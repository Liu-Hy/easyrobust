"""Single-node training script on the HAL server"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import SubsetRandomSampler
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchattacks
#from torchvision import transforms
from ImageNetDG_l import ImageNetDG_l
from timm.utils import ModelEmaV2, distribute_bn
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def adv_train(dataloader, model, criterion, optimizer, scheduler, args, delta_x, train_ratio, epoch, model_ema):
    model.train()
    if args.adv == 'madry':
        attack = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True)
    elif args.adv == 'trades':
        attack = torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=10)
        criterion_kl = nn.KLDivLoss(size_average=False)
    for step, batch in enumerate(dataloader):
        if step > int(train_ratio * len(dataloader)):
            break
        imgs, labels = [x.cuda(non_blocking=True) for x in batch]
        optimizer.zero_grad()
        outputs = model(imgs)
        # optimizer.zero_grad()
        loss = criterion(outputs, labels)
        if args.adv == 'madry':
            adv_imgs = attack(imgs, labels)
            adv_outputs = model(adv_imgs)
            adv_loss = criterion(adv_outputs, labels)
            loss = loss + args.beta * adv_loss
        elif args.adv == 'trades':
            adv_imgs = attack(imgs, labels)
            adv_outputs = model(adv_imgs)
            adv_loss = (1.0 / args.batch_size) * criterion_kl(F.log_softmax(adv_outputs, dim=1),
                                                            F.softmax(outputs, dim=1))
            loss = loss + args.beta * adv_loss
        if args.ns:
            x = model.module[1].patch_embed(model.module[0](imgs))
            x = x + delta_x.cuda(non_blocking=True)
            ns_outputs = model.module[1].head(encoder_forward(model, x))
            ns_loss = criterion(ns_outputs, labels)
            # consistency = ((ns_outputs - outputs) ** 2).sum(dim=-1).mean()
            loss = loss + ns_loss  # + consistency
        loss.backward()
        optimizer.step()
        scheduler.step()
        if model_ema is not None:
            model_ema.update(model)
        if step % 500 == 0:
            print(
                f'Epoch: {epoch}, Step {step}, Loss: {round(loss.item(), 4)}', flush=True)


def validate(dataloader, model, criterion, val_ratio, adv=False):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    if adv:
        attack = torchattacks.FGSM(model, eps=1 / 255)
    for step, batch in enumerate(dataloader):
        if step > int(val_ratio * len(dataloader)):
            break
        samples, labels = [x.cuda(non_blocking=True) for x in batch]
        if adv:
            samples = attack(samples, labels)
        with torch.no_grad():
            outputs = model(samples)
            # print(f'output shape: {outputs.shape}')
            loss += criterion(outputs, labels)
            _, preds = outputs.topk(5, -1, True, True)
            correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
            correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
            total += samples.size(0)

    loss_val = loss.item() / len(dataloader)
    acc1 = 100 * correct1.item() / total.item()
    acc5 = 100 * correct5.item() / total.item()

    return acc1, loss_val


def validate_corruption(data_path, info_path, model, transform, criterion, batch_size, val_ratio):
    result = dict()
    type_errors = []
    for typ in tqdm(CORRUPTIONS, position=0, leave=True):
        type_path = data_path.joinpath(typ)
        assert type_path in list(data_path.iterdir())
        errors = []
        for s in range(1, 6):
            s_path = type_path.joinpath(str(s))
            assert s_path in list(type_path.iterdir())
            loader = prepare_loader(s_path, info_path, batch_size, transform)
            acc, _ = validate(loader, model, criterion, val_ratio)
            errors.append(100 - acc)
        type_errors.append(get_mean(errors))
    me = get_mean(type_errors)
    relative_es = [(e / al) for (e, al) in zip(type_errors, ALEX)]
    mce = 100 * get_mean(relative_es)
    result["es"] = type_errors
    result["ces"] = relative_es
    result["me"] = me
    result["mce"] = mce
    print(f"mCE: {mce:.2f}%, mean_err: {me}%", flush=True)
    return result


def prepare_loader(split_data, info_path, batch_size, transform=None):
    if isinstance(split_data, (str, Path)):
        split_data = ImageNetDG_l(split_data, info_path, transform=transform)
    data_loader = DataLoader(split_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return data_loader


def main(args):
    # run = wandb.init(project="nullspace", group="hal")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 超参数设置
    epochs = args.epochs
    train_batch_size = args.batch_size
    val_batch_size = int(train_batch_size * 1.5)
    rounds = 3
    lr = args.lr  # When using SGD and StepLR, set to 0.001
    lim = args.lim
    nlr = args.nlr
    eps = args.eps
    if args.ns_mode != 'default':
        assert args.ns is True, 'When nullspace mode is set, nullspace training must be turned on'
    img_ratio = 0.1  # 0.02
    train_ratio = 0.004  # 0.1
    val_ratio = 1.  # 0.05
    save_path = Path(args.output)
    data_path = Path(args.data_dir)  # Path("/var/lib/data")
    save_path.mkdir(exist_ok=True, parents=True)

    if args.debug:
        train_batch_size, val_batch_size = 2, 2
        img_ratio, train_ratio, val_ratio = 0.001, 0.001, 0.1

    # 模型、数据、优化器
    model_name = args.model
    #ckpt_path = args.resume
    ckpt_path = './best_epoch'
    model, patch_size, img_size, model_config = get_model_and_config(model_name, ckpt_path=ckpt_path, use_ema=False)

    normalize = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model)

    model.cuda()
    # Wrap the model
    model = nn.DataParallel(model)
    model_ema = None
    if args.use_ema:
        model_ema = ModelEmaV2(model, decay=0.9998, device=None)
        load_checkpoint(model_ema.module.module, ckpt_path, use_ema=True)

    m = model_name.split('_')[1]
    setting = f'{m}_ps{patch_size}_epochs{epochs}_lr{lr}_bs{train_batch_size}_ns_{args.ns}_mode_{args.ns_mode}_adv_{args.adv}_nlr{nlr}_rounds{rounds}' + \
              f'_lim{lim}_eps{eps}_imgr{img_ratio}_trainr{train_ratio}_valr{val_ratio}'
    setting_path = save_path.joinpath(setting)
    noise_path = setting_path.joinpath("noise")
    model_path = setting_path.joinpath("model")
    setting_path.mkdir(exist_ok=True, parents=True)
    noise_path.mkdir(exist_ok=True, parents=True)
    model_path.mkdir(exist_ok=True, parents=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])

    info_path = Path("info")

    held_out = 0.1
    data_set = ImageNetDG_l(data_path.joinpath('imagenet/train'), info_path, train_transform)
    len_dev = int(held_out * len(data_set))
    len_train = len(data_set) - len_dev
    train_set, dev_set = torch.utils.data.random_split(data_set, (len_train, len_dev))
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    img_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    dev_loader = prepare_loader(dev_set, info_path, val_batch_size)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(model_config['mean'], model_config['std'])])
    typ_path = data_path.joinpath("imagenet", "val")
    val_loader = prepare_loader(typ_path, info_path, val_batch_size, val_transform)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    best_acc = 0.

    # 训练、验证
    start_epoch = len(list(model_path.iterdir()))
    if start_epoch > 0:
        print(f"Restore training from epoch {start_epoch}")
        checkpoint = torch.load(model_path.joinpath(str(start_epoch - 1)))
        model.module.load_state_dict(checkpoint["model_state_dict"])
        if args.use_ema:
            model_ema.module.module.load_state_dict(checkpoint["state_dict_ema"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_acc = torch.load(setting_path.joinpath("best_epoch"))["best_acc"]
        print(f"Previous best acc: {best_acc}")

    model_to_validate = model_ema.module if args.use_ema else model
    result = dict()
    val_acc, _ = validate(val_loader, model_to_validate, criterion, val_ratio)
    result["val"] = val_acc
    print(result)
    val_acc, _ = validate(val_loader, model_to_validate, criterion, val_ratio, adv=True)
    result["fgsm"] = val_acc
    total = get_mean([100 - v if k == "corruption" else v for k, v in result.items()])
    print(f"Avg performance: {total}\n", result)

    delta_x = None
    for epoch in range(start_epoch, epochs):
        if args.ns:
            if Path.exists(noise_path.joinpath(str(epoch))):
                print(f"Loading learned noise at epoch {epoch}")
                delta_x = torch.load(noise_path.joinpath(str(epoch)))['delta_x']
            else:
                print("---- Learning noise")
                delta_x = encoder_level_epsilon_noise(model, img_loader, img_size, rounds, nlr, lim, eps, img_ratio, ns_mode=args.ns_mode)
                torch.save({"delta_x": delta_x}, noise_path.joinpath(str(epoch)))
            print(f"Noise norm: {round(torch.norm(delta_x).item(), 4)}")

        print("---- Training model")
        adv_train(train_loader, model, criterion, optimizer, scheduler, args, delta_x, train_ratio, epoch, model_ema)

        if (epoch+1) % 4 == 0:
            print("---- Validating model")
            ema_to_save = model_ema.module.module.state_dict() if args.use_ema else None
            result = dict()
            # Evaluate on held-out set
            dev_acc, _ = validate(dev_loader, model_to_validate, criterion, val_ratio)
            # Evaluate on val set
            val_acc, _ = validate(val_loader, model_to_validate, criterion, val_ratio)
            result["val"] = val_acc
            print(result)
            val_acc, _ = validate(val_loader, model_to_validate, criterion, val_ratio, adv=True)
            result["fgsm"] = val_acc
            torch.save({"model_name": model_name, "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "state_dict_ema": ema_to_save,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "result": result}, model_path.joinpath(str(epoch)))
            # 保存
            print(f"Dev acc: {dev_acc}")
            total = get_mean([100 - v if k == "corruption" else v for k, v in result.items()])
            print(f"Avg performance: {total}\n", result)
            if dev_acc > best_acc:
                best_acc = dev_acc
                print(f'New Best Acc: {best_acc:.2f}%')
                torch.save({"model_state_dict": model.module.state_dict(), "best_epoch": epoch, "best_acc": best_acc},
                           setting_path.joinpath("best_epoch"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('--ns', action='store_true')
    parser.add_argument('--ns-mode', choices=['default', 'random'],
                        default='default', help='Options for nullspace noise generation')
    parser.add_argument('--use-ema', action='store_true',
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--adv', choices=['trades', 'madry', 'none'],
                        default='none', help='Adversarial attack options')

    parser.add_argument('--model', default='vit_small_patch16_224', type=str,
                        help='Name of model to train (default: "resnet50"')
    parser.add_argument('--resume', default='../../../../pretrained/vit_small_patch16_224.npz', type=str,
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--data_dir', default='../../../../../data',
                        help='path to dataset')
    parser.add_argument('--output', default='../output/hal', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')

    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for updating network')
    parser.add_argument('--lim', type=float, default=3, help='sampling limit of the noise')
    parser.add_argument('--nlr', type=float, default=0.1, help='learning rate for the noise')
    parser.add_argument('--eps', type=float, default=0.03, help='threshold to stop training the noise')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--beta', type=float, default=1.0, help='coefficient for adversarial training')
    #parser.add_argument('--no-adv', action='store_true')



    args = parser.parse_args()
    main(args)
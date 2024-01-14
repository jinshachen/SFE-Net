import os
import time
import datetime
import torch
import torch.nn as nn

from Network.SFNet import SFNet

from train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import CellDataset
import transforms as T


class SegmentationPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5):
        trans = []
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.append(T.RandomRotation(90))
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class Norm:
    def __init__(self):
        self.Norm = T.Compose([
            T.ToTensor(),
        ])
    
    def __call__(self, img, target):
        return self.Norm(img, target)


def create_model(num_classes):
    model = SFNet(in_channels=3, num_classes=num_classes)
    return model


def main(args):
    dataset = args.dataset
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using device is: ", device)
    batch_size = args.batch_size
    print('batch_size:', batch_size)
    num_classes = args.num_classes + 1

    train_dataset = CellDataset(args.train_data,
                                 train=True,
                                 transforms=SegmentationPresetTrain(), Norm=Norm())

    val_dataset = CellDataset(args.test_data,
                               train=False,
                               Norm=Norm())

    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    save_feat = False
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes, save_feat,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        confmat, dice, test_loss = evaluate(model, val_loader, device=device, num_classes=num_classes, save_feat=save_feat)

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
                save_feat = True
            else:
                save_feat = False
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "./save_weights/best_model_{}.pth".format(dataset))
        else:
            torch.save(save_file, "./save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    print("best dice: ", best_dice)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--train-data", default="./TNBC256")
    parser.add_argument("--test-data", default="./TNBC256")
    parser.add_argument("--dataset", default="TNBC256", help="Dataset name")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')  # ori: 0.001
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)

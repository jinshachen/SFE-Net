import torch
from torch import nn
import train_utils.distributed_utils as utils
from train_utils.dice_coefficient_loss import dice_loss, build_target
from PreSFNetV1 import LGD


def FeatureLoss(F1, F2):
    loss_func = nn.MSELoss()
    loss = loss_func(F1, F2)
    return loss


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True):
    loss = nn.functional.cross_entropy(inputs, target, weight=loss_weight)

    if dice is True:
        dice_target = build_target(target, num_classes)
        d_loss = dice_loss(inputs, dice_target, multiclass=True)
        loss = loss + d_loss

    return loss


def evaluate(model, data_loader, device, num_classes, save_feat):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target, img_path in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output, x1, x2, x3, x4, x5, o1, o2, o3, o4 = model(image)
            test_loss = criterion(output, target, num_classes=num_classes)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item(), test_loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes, save_feat, 
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for image, target, img_path in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            mask = target.unsqueeze(1).float()
            teacher_model = LGD()
            with torch.no_grad():
                logits, f1, f2, f3, f4, f5, l1, l2, l3, l4 = teacher_model(mask)
            output, x1, x2, x3, x4, x5, d1, d2, d3, d4 = model(image)

            loss = criterion(output, target, loss_weight, num_classes=num_classes)

            feature_loss3 = FeatureLoss(d3, l3)
            feature_loss4 = FeatureLoss(d4, l4)
            a = 0.5
            loss = loss + a * (feature_loss3 + feature_loss4)  # distill

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

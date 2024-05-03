import config as c
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler
import random
import torch.backends.cudnn as cudnn
import argparse
import warnings
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.log_helper import init_logger, pre_log
from utils.utils import model_setting, data_gen


# for command line
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--cuda', default=True, type=bool, metavar='N', help='Specify if cuda')
    parser.add_argument('--model', default="Reset3D10Attention", type=str, metavar='N',
                        help='Reset3D10Attention, Reset3D18Attention')
    # In most datasets, the performance of Reset3D10Attention is better than Reset3D18Attention
    parser.add_argument('--normalize', default=True, type=bool, metavar='N', help='normalizing input data')
    parser.add_argument('--combine', default=False, type=bool, metavar='N', help='combining right and left part')
    parser.add_argument('--augmentation', default=True, type=bool, metavar='N',
                        help='data augmentation(only for model training)')
    args = parser.parse_args()
    return args


def train(config, train_loader, model, criterion, optimizer, scheduler, device):
    model.train()
    Scaler = torch.cuda.amp.GradScaler()
    for img, label, _ in tqdm(train_loader):
        img = img.to(device)
        label = label.to(device)
        with autocast(enabled=config['autocast']):
            output = model(img)
            loss = criterion(output, label)
        # compute gradient and do optimizing step
        Scaler.scale(loss).backward()
        Scaler.step(optimizer)
        Scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    return OrderedDict([('loss', loss)])


def validate(config, val_loader, model, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for img, label, _ in tqdm(val_loader):
            img = img.to(device)
            label = label.to(device)

            with autocast(enabled=config['autocast']):
                output = model(img)
                loss = criterion(output, label)
                output = torch.sigmoid(output)
                # total loss
                total_loss += loss.item()
            predicted = (output > threshold).float()

            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    return OrderedDict([('accuracy', accuracy), ('loss', average_loss)])


def seed_and_settings(seed_value=46):
    args = parse_args()
    # pre setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = c.__config__
    warnings.filterwarnings("ignore")
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    if not os.path.exists(config['model_save_dir']):
        os.makedirs(config['model_save_dir'])

    # seed everything
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_ordinal)
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    return device, config, args


def main():
    device, config, args = seed_and_settings()
    Logger = init_logger(log_file=config['log_path'])
    pre_log(Logger, args)
    model, optimizer, criterion, scheduler = model_setting(args, config, device, mode='train')
    train_loader, val_loader = data_gen(config, args, mode='train')

    best_val_loss = np.inf
    best_val_acc = 0

    for epoch in range(config['epochs']):
        epoch = epoch + 1
        Logger.info('Epoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, train_loader, model, criterion, optimizer, scheduler, device)

        if epoch % config['val_epoch'] == 0 and epoch > config['start_val']:
            val_log = validate(config, val_loader, model, criterion, device, threshold=config['threshold'])
            Logger.info(
                f'Epoch {epoch} - avg_train_loss: {train_log["loss"]:.4f}  avg_val_loss: {val_log["loss"]:.4f}  '
                f'val_accuracy: {val_log["accuracy"]:.4f}')

            if val_log['loss'] < best_val_loss or val_log["accuracy"] > best_val_acc:
                path = os.path.join(config['model_save_dir'],
                                    f'{args.model}_epoch%s_acc{round(val_log["accuracy"], 4)}.pth' % epoch)
                torch.save(model.state_dict(), path)
                best_val_loss = min(val_log['loss'], best_val_loss)
                best_val_acc = max(val_log["accuracy"], best_val_acc)
                Logger.info("\n===============> Saved best model<===============")
                Logger.info(f'Best val loss:{val_log["loss"]:.4f}      Best accuracy{val_log["accuracy"]:.4f}')
                Logger.info("===============> Saved best model<===============\n")
        else:
            Logger.info(
                f'Epoch {epoch} - avg_train_loss: {train_log["loss"]:.4f}')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
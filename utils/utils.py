import torch.nn as nn
from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Reset3DAttention


def model_setting(args, config, device, mode='train'):
    # create model
    if args.model == 'baseline_model':
        pass
    elif args.model == 'Reset3D18Attention':
        model = Reset3DAttention(model_depth=18)
    elif args.model == 'Reset3D10Attention':
        model = Reset3DAttention(model_depth=10)
    else:
        raise NotImplementedError

    # model.to(device)
    if mode == 'train':
        params = filter(lambda p: p.requires_grad, model.parameters())

        # optimizer / criterion / scheduler
        optimizer = torch.optim.Adam(params, config['learning_rate'], weight_decay=config['weight_decay'])
        criterion = nn.BCEWithLogitsLoss()
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        else:
            raise NotImplementedError
        model.to(device)
        return model, optimizer, criterion, scheduler
    elif mode == 'test':
        model.load_state_dict(torch.load(args.model_path))
        return model
    else:
        raise NameError


def data_gen(config, args, mode='train'):
    if mode == 'train':
        train_dataset = Dataset(input_size=config['img_dim'], dataset_path=config['train_data_path'],
                                normalize=args.normalize, combine=args.combine, augmentation=args.augmentation)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=config['num_workers'],
                                  drop_last=True,
                                  pin_memory=True)
        val_dataset = Dataset(input_size=config['img_dim'], dataset_path=config['val_data_path'],
                              normalize=args.normalize, combine=args.combine, augmentation=False)
        val_loader = DataLoader(val_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers=config['num_workers'],
                                drop_last=False,
                                pin_memory=True)
        return train_loader, val_loader
    elif mode == 'test':
        test_dataset = Dataset(input_size=config['img_dim'], dataset_path=args.test_path, normalize=args.normalize,
                               combine=args.combine, augmentation=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=8,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=False,
                                 pin_memory=True)
        return test_loader
    elif mode == 'Xai':
        test_dataset = Dataset(input_size=config['img_dim'], dataset_path=args.test_path, normalize=args.normalize,
                               combine=args.combine, augmentation=False, Xai=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=8,
                                 drop_last=False,
                                 pin_memory=True)
        return test_loader
    else:
        raise NameError

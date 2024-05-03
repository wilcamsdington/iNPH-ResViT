__config__ = {
    # dataset / logger/ model path
    'train_data_path': r'C:\Users\13459\Desktop\iNPH\data\fastsurfer_seg_nii_train',
    'val_data_path': r'C:\Users\13459\Desktop\iNPH\data\fastsurfer_seg_nii_val',
    'log_path': r'Trained model/log.txt',
    'model_save_dir': r'Trained model',

    # Model
    'img_dim': 224,  # img input size (img_dim * img_dim * img_dim)
    'num_workers': 8,
    'autocast': True,
    'threshold': 0.5,

    # Training
    'epochs': 100,
    'batch_size': 4,  # val's batch size is 2 * batch_size
    'weight_decay': 1e-5,
    'scheduler': 'CosineAnnealingLR',
    'min_lr': 1e-6,  # min learning rate for scheduler
    'learning_rate': 1e-4,
    'val_epoch': 1,  # Validate the model in val data every "config['val_epoch']" epoch
    'start_val': 24,  # After "config['start_val']+1" epoch, model will start val in val dataset
}
from sklearn.metrics import classification_report, precision_recall_fscore_support
import config as c
from torch.cuda.amp import autocast, GradScaler
import random
import torch.backends.cudnn as cudnn
import argparse
import warnings
import torch
import csv
import os
import numpy as np
from tqdm import tqdm
from utils.utils import model_setting, data_gen


# for command line
def parse_args():
    """
    make sure model and model_path is the same model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--cuda', default=True, type=int, metavar='N', help='Specify if cuda')
    parser.add_argument('--model', default="Reset3D10Attention", type=str, metavar='N', help='Reset3D10Attention, Reset3D18Attention')

    # change test data set path and model path
    parser.add_argument('--test_path', default=r'data\fastsurfer_seg_nii_test', type=str, metavar='N', help='test dir')
    parser.add_argument('--model_path', default=r"Trained model\Reset3D10Attention.pth",
                        type=str, metavar='N')
    parser.add_argument('--normalize', default=True, type=bool, metavar='N', help='normalizing input data')
    parser.add_argument('--combine', default=False, type=bool, metavar='N', help='combining right and left part FF seg')
    parser.add_argument('--predict_only', default=True, type=bool, metavar='N', help='if no label provide, set it True')

    # if you want to test ensemble_model, change the following argument or you can ignore it.
    parser.add_argument('--ensemble', default=False, type=bool, metavar='N', help='if use ensemble model')
    parser.add_argument('--ensemble_model', default=['Reset3D18Attention',
                                                     'Reset3D18Attention',
                                                     'Reset3D18Attention'],
                        type=list, metavar='N', help='list of ensemble model')
    parser.add_argument('--ensemble_model_path', default=['Trained model/Reset3D18Attention_epoch46_acc0.9778.pth',
                                                          'Trained model/Reset3D18Attention_epoch41_acc0.9556.pth',
                                                          'Trained model/Reset3D18Attention_epoch31_acc0.9556.pth'],
                        type=list, metavar='N', help='list of ensemble model path')
    args = parser.parse_args()
    return args


def test(config, args, model, device, threshold=0.5, save_csv=True):
    print('\nTesting model:', args.model)
    print('Loading model from:', args.model_path)
    print('Setting threshold:', threshold)
    print('testing data path:', args.test_path)
    model.to(device)
    model.eval()
    all_labels = []
    all_predictions = []
    preds_dic = {}
    test_loader = data_gen(config, args, mode='test')

    with torch.no_grad():
        for img, label, case_name in tqdm(test_loader):
            img = img.to(device)
            # label = label.to(device)

            with autocast(enabled=config['autocast']):
                output = model(img)
                output = torch.sigmoid(output)
            predicted = (output > threshold).float().cpu().numpy()
            predict_probs = output.float().cpu().numpy()
            all_labels.append(label.cpu().numpy())
            if save_csv:
                case_name = list(case_name)
                predicted = list(predicted)
                predict_probs = list(predict_probs)
                for i in range(len(case_name)):
                    preds_dic[case_name[i]] = (predicted[i][0], predict_probs[i][0])
            all_predictions.append(predicted)
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    print(classification_report(all_labels, all_predictions, digits=4))
    print("Recall for 1 is Sensitivity and Recall for 0 is Specificity.")
    if save_csv:
        return preds_dic
    else:
        return None


def ensemble_test(args, config, device, threshold=0.5):
    assert args.ensemble_model is not None
    assert args.ensemble_model_path is not None
    assert len(args.ensemble_model) == len(args.ensemble_model_path)
    num_model = len(args.ensemble_model)

    test_loader = data_gen(config, args, mode='test')
    print('Ensemble model predicting...')
    preds_dic = {}
    for i in range(num_model):
        args.model = args.ensemble_model[i]
        args.model_path = args.ensemble_model_path[i]
        print(f'\nModel{i + 1}:{args.model}\nPath:{args.model_path} \nPredicting...')
        model = model_setting(args, config, device, mode='test')
        model.to(device)
        model.eval()
        with torch.no_grad():
            for img, _, case_name in tqdm(test_loader):
                img = img.to(device)
                with autocast(enabled=config['autocast']):
                    output = model(img)
                    output = torch.sigmoid(output)
                output = output.cpu().numpy()
                case_name = list(case_name)
                output = list(output)
                for i_bs in range(len(case_name)):
                    if i == 0:
                        preds_dic[case_name[i_bs]] = output[i_bs][0]
                    else:
                        preds_dic[case_name[i_bs]] += output[i_bs][0]
    # Averaging probability
    threshold = threshold * num_model
    for case in preds_dic.keys():
        if preds_dic[case] > threshold:
            preds_dic[case] = (1.0, preds_dic[case] / num_model)
        else:
            preds_dic[case] = (0.0, preds_dic[case] / num_model)
    if not args.predict_only:
        all_labels = []
        all_predictions = []
        for key, value in preds_dic.items():
            all_predictions.append(value[0])
            if 'impr' in key:
                all_labels.append(1.0)
            elif 'contr' in key:
                all_labels.append(0.0)
            else:
                raise NameError
        print(classification_report(all_labels, all_predictions, digits=4))
        print("Recall for 1 is Sensitivity and Recall for 0 is Specificity.")
    return preds_dic


def seed_and_settings_test(seed_value=46):
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


def save_csv(preds_dic, args):
    if preds_dic is not None:
        if not os.path.exists('results'):
            os.makedirs('results')
        csv_file_path = 'results\\ensemble_model_preds.csv' if args.ensemble else f'results\\{args.model}_preds.csv'
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['case_name', 'predict', 'probability']
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for key, value in preds_dic.items():
                csv_writer.writerow({'case_name': key, 'predict': value[0], 'probability':value[1]})
        print(f'CSV file have been created：{csv_file_path}')


def main():
    device, config, args = seed_and_settings_test()
    if args.ensemble:
        preds_dic = ensemble_test(args, config, device)
    else:
        model = model_setting(args, config, device, mode='test')
        preds_dic = test(config, args, model, device, threshold=0.5)
    save_csv(preds_dic, args)


if __name__ == '__main__':
    main()

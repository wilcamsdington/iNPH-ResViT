"""
It is an example code of Xai we used in paper
"""
import matplotlib.pyplot as plt
import config as c
import random
import nibabel as nib
import torch.backends.cudnn as cudnn
import argparse
import warnings
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.utils import model_setting, data_gen
from medcam.medcam import inject
from scipy.ndimage import gaussian_filter


# for command line
def parse_args():
    """
    make sure model and model_path is the same model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_ordinal', default=0, type=int, metavar='N', help='Specify which graphics card to use')
    parser.add_argument('--cuda', default=True, type=int, metavar='N', help='Specify if cuda')

    # change test data set path and model path
    parser.add_argument('--model', default="Reset3D10Attention", type=str, metavar='N', help='baseline_model, resnet18')
    parser.add_argument('--test_path', default=r"data\fastsurfer_seg_nii_val", type=str, metavar='N', help='test dir')
    parser.add_argument('--model_path', default=r"Trained model\Reset3D10Attention.pth",
                        type=str, metavar='N')
    parser.add_argument('--normalize', default=True, type=bool, metavar='N', help='normalizing input data')
    parser.add_argument('--combine', default=False, type=bool, metavar='N', help='combining right and left part')

    # change as needed
    parser.add_argument('--test_case_name', default=['case1', 'case2', 'case3'], type=list, metavar='N',
                        help='choose which case you want to test, you can test multiple cases')
    parser.add_argument('--run_all', default=False, type=bool, metavar='N',
                        help='if you want to run all cases in test_path, set it True')
    parser.add_argument('--visualisation', default=False, type=bool, metavar='N', help='to visualize some slices')
    parser.add_argument('--save_as_nii', default=False, type=bool, metavar='N', help='to save heatmap and img as nii.gz')
    parser.add_argument('--nii_dir', default='XAI_results', type=str, metavar='N', help='dir to save')
    args = parser.parse_args()
    return args


def test_Xai(config, args, model, device, threshold=0.5):
    print('\nTesting model:', args.model)
    print('Loading model from:', args.model_path)
    print('Setting threshold:', threshold)
    print('testing data path:', args.test_path)

    # check name for different layers of model
    # for name, module in model.named_modules():
    #     print(name)
    """
    We selected the 'resnet.layer4.0.conv2' layer as an example, but you can choose different layers to analyze as well.
    You can find detailed of the XAI techniques we employed in the following GitHub repository:
    https://github.com/MECLabTUDA/M3d-Cam 
    """
    model.to(device)
    model = inject(model, output_dir="XAI", save_maps=False, backend='gcam', replace=True,
                   layer=[('resnet.layer4.0.conv2')])
    model.eval()
    test_loader = data_gen(config, args, mode='Xai')

    with torch.no_grad():
        for img, mask, label, case_name, orig_nu in tqdm(test_loader):
            if case_name[0] in args.test_case_name or args.run_all:
                img = img.to(device)
                mask = mask.to(device)
                orig_nu = orig_nu.to(device)
                output = model(img)
                output = output.squeeze().numpy()

                mask = mask.squeeze().numpy()
                output = output * mask
                output = output / output.max()
                output = gaussian_filter(output, sigma=8)
                orig_nu = orig_nu.squeeze().numpy()

                if args.visualisation:
                    Visual(output, orig_nu)
                if args.save_as_nii:
                    if not os.path.exists(args.nii_dir):
                        os.makedirs(args.nii_dir)
                    save_nii(os.path.join(args.nii_dir, f'{case_name[0]}_heatmap.nii.gz'), output)
                    save_nii(os.path.join(args.nii_dir, f'{case_name[0]}_orig_nu.nii.gz'), orig_nu)
                    print(f'Case:{case_name[0]} saved!')
            else:
                continue


def seed_and_settings_test(seed_value=46):
    args = parse_args()
    # pre setting
    device = torch.device('cuda' if args.cuda else 'cpu')
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
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.set_device(args.cuda_ordinal)
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    return device, config, args


def Visual(heatmap, img):
    assert heatmap.shape == img.shape
    data = heatmap
    data1 = img
    shape = data1.shape
    for i in range(shape[0]//2-15, shape[0]//2+15, 5):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

        slice_heatmap = data[:, i, :]
        slice_heatmap1 = data1[:, i, :]
        slice_heatmap = np.rot90(slice_heatmap)
        slice_heatmap1 = np.rot90(slice_heatmap1)
        axes[0].imshow(slice_heatmap)
        axes[0].imshow(slice_heatmap1, cmap='gray', alpha=0.45)
        axes[0].set_title('Axial')
        axes[0].axis('off')

        slice_heatmap = data[i, :, :]
        slice_heatmap1 = data1[i, :, :]
        axes[1].imshow(slice_heatmap)
        axes[1].imshow(slice_heatmap1, cmap='gray', alpha=0.45)
        axes[1].set_title('Sagittal')
        axes[1].axis('off')

        slice_heatmap = data[:, :, i]
        slice_heatmap1 = data1[:, :, i]
        slice_heatmap = np.rot90(slice_heatmap, k=3)
        slice_heatmap1 = np.rot90(slice_heatmap1, k=3)
        axes[2].imshow(slice_heatmap)
        axes[2].imshow(slice_heatmap1, cmap='gray', alpha=0.45)
        axes[2].set_title('Coronal')
        axes[2].axis('off')
        plt.suptitle(f'Slice:{i}', fontsize=16)
        plt.tight_layout()
        plt.show()
        # change name here!
        # plt.savefig(f'{i}.svg', format='svg')
        # plt.savefig(f'{i}.png', format='png')


def save_nii(file_name, output):
    nifti_image_mri = nib.Nifti1Image(output*255, affine=np.eye(4))
    nib.save(nifti_image_mri, file_name)


def main():
    device, config, args = seed_and_settings_test()
    model = model_setting(args, config, device, mode='test')
    test_Xai(config, args, model, device, threshold=0.5)


if __name__ == '__main__':
    main()

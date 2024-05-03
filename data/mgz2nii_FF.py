"""generate the dataset for the model"""
import nibabel as nib
import os
from tqdm import tqdm


main_path = "Small NPH dataset FS output"
path_list = os.listdir(main_path)
output_path = "Small NPH dataset FS output_nii"
SS_mask = r"C:\Users\13459\Desktop\iNPH\data\TIVmasks"

"""
FF output -> dataset
"""

for path in tqdm(path_list):
    print(path)
    name = path
    path = path+'\\mri'
    path = os.path.join(main_path, path)

    # result mgz2nii
    imput_file_name = 'aparc.DKTatlas+aseg.deep.mgz'
    input_mgz_file = os.path.join(path, imput_file_name)
    file_name = 'aparc.DKTatlas+aseg.deep.nii.gz'
    output_nii_file = os.path.join(output_path, name)
    os.makedirs(output_nii_file, exist_ok=True)
    img = nib.load(input_mgz_file)
    nib.save(img, os.path.join(output_nii_file, file_name))

    """
    imput_file_name = 'mask.mgz'
    input_mgz_file = os.path.join(path, imput_file_name)
    file_name = 'mask.nii.gz'
    output_nii_file = os.path.join(output_path, name)
    img = nib.load(input_mgz_file)
    nib.save(img, os.path.join(output_nii_file, file_name))
    """

    """
    imput_file_name = 'orig.mgz'
    input_mgz_file = os.path.join(path, imput_file_name)
    file_name = 'orig.nii.gz'
    output_nii_file = os.path.join(output_path, name)
    img = nib.load(input_mgz_file)
    nib.save(img, os.path.join(output_nii_file, file_name))
    """

    # biasfield-corrected image
    imput_file_name = 'orig_nu.mgz'
    input_mgz_file = os.path.join(path, imput_file_name)
    file_name = 'orig_nu.nii.gz'
    output_nii_file = os.path.join(output_path, name)
    img = nib.load(input_mgz_file)
    nib.save(img, os.path.join(output_nii_file, file_name))

    # Add more data if you want to use in model training
    # SS mask
    imput_file_name = name+'_mask.mgz'
    input_mgz_file = os.path.join(SS_mask, imput_file_name)
    file_name = 'SS_mask.nii.gz'
    output_nii_file = os.path.join(output_path, name)
    img = nib.load(input_mgz_file)
    nib.save(img, os.path.join(output_nii_file, file_name))
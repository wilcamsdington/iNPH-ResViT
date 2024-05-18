import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
from torchvision.transforms import v2


def crop_volume(volume, input_size=256):
    target_size = (input_size, input_size, input_size)
    center_x = volume.shape[0] // 2
    center_y = volume.shape[1] // 2
    center_z = volume.shape[2] // 2

    start_x = center_x - target_size[0] // 2
    end_x = start_x + target_size[0]

    start_y = center_y - target_size[1] // 2
    end_y = start_y + target_size[1]

    start_z = center_z - target_size[2] // 2
    end_z = start_z + target_size[2]

    cropped_volume = volume[start_x:end_x, start_y:end_y, start_z:end_z]
    return cropped_volume


class Dataset(Dataset):
    def __init__(self, input_size=224, dataset_path=None, normalize=True, combine=True,
                 augmentation=True, Xai=False):
        assert dataset_path is not None
        self.input_size = input_size
        self.dataset_path = dataset_path
        # self.mir_list = ['aparc.DKTatlas+aseg.deep.nii.gz', 'mask.nii.gz', 'orig_nu.nii.gz']
        self.mir_list = ['aparc.DKTatlas+aseg.deep.nii.gz', 'SS_mask.nii.gz', 'orig_nu.nii.gz']
        # ['aparc.DKTatlas+aseg.deep.nii.gz', 'mask.nii.gz', 'orig.nii.gz', 'orig_nu.nii.gz']
        self.path_list = self.load_dir()
        self.normalize = normalize
        self.combine = combine
        self.Xai = Xai
        if augmentation:
            self.data_transform = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=(-20, 20), translate=(0.05, 0.05), scale=[0.95, 1.05])])
        else:
            self.data_transform = None

    def load_dir(self):
        path_list = []
        name_list = os.listdir(self.dataset_path)
        for name in name_list:
            path_list.append(os.path.join(self.dataset_path, name))
        return path_list

    def __len__(self):
        return len(self.path_list)

    def load_data(self, path):
        # loading data
        data_list = []
        for i in self.mir_list:
            data_path = os.path.join(path, i)
            img_data = nib.load(data_path)
            img = img_data.get_fdata()
            if img.shape[0] > self.input_size:
                img = crop_volume(img, input_size=self.input_size)
            data_list.append(img)

        # label
        case_name = os.path.basename(path)
        if 'impr' in case_name:
            label = torch.tensor([1.])
        elif 'contr' in case_name:
            label = torch.tensor([0.])
        else:
            label = torch.tensor([-1.])  # no label

        seg, mask, orig_nu = data_list

        # to tensor
        seg = torch.tensor(seg).to(torch.float)  # torch.Size([224, 224, 224])
        mask = torch.tensor(mask).to(torch.float)  # torch.Size([224, 224, 224])
        orig_nu = torch.tensor(orig_nu).to(torch.float)  # torch.Size([224, 224, 224])
        return case_name, label, seg, mask, orig_nu

    def combining_func(self, seg):
        if self.combine:
            def right_left(current_item):
                """
                FS_dict = {41: 2, 43: 4, 44: 5, 46: 7, 47: 8, 49: 10, 50: 11, 51: 12, 52: 13, 53: 17, 54: 18, 58: 26,
                           60: 28, 63: 31,
                           2002: 1002, 2003: 1003, 2005: 1005, 2006: 1006, 2007: 1007, 2008: 1008, 2009: 1009, 2010: 1010,
                           2011: 1011, 2012: 1012, 2013: 1013, 2014: 1014, 2015: 1015, 2016: 1016, 2017: 1017, 2018: 1018,
                           2019: 1019, 2020: 1020, 2021: 1021, 2022: 1022, 2023: 1023, 2024: 1024, 2025: 1025, 2026: 1026,
                           2027: 1027, 2028: 1028, 2029: 1029, 2030: 1030, 2031: 1031, 2034: 1034, 2035: 1035}
                """
                FS_dict = {41: 2, 43: 4, 44: 5, 46: 7, 47: 8, 49: 10, 50: 11, 51: 12, 52: 13, 53: 17, 54: 18, 58: 26,
                           60: 28, 63: 31}
                return FS_dict.get(current_item, current_item)

            # For Subcortical Structures
            seg.apply_(right_left)
            # For Cortical Structures
            seg[seg > 2000] -= 1000
        return seg

    def normalize_func(self, seg, orig_nu):
        if self.normalize:
            if not self.combine:
                seg[seg > 2000] -= 1900
            seg[seg > 1000] -= 900  # max_value = 135.0
            seg = seg / 135.0
            orig_nu = orig_nu / 255.0
        return seg, orig_nu

    def preprocessing(self, path):
        case_name, label, seg, mask, orig_nu = self.load_data(path)
        """preprocessing"""
        seg = self.combining_func(seg)  # combining right and left part
        seg, orig_nu = self.normalize_func(seg, orig_nu)  # Normalization
        orig_nu_o = orig_nu
        orig_nu = orig_nu * mask
        img = torch.cat([seg.unsqueeze(0), orig_nu.unsqueeze(0)], dim=0)
        if self.data_transform is not None:
            img = self.data_transform(img)  # data augmentation
        if self.Xai:
            return img, mask, label, case_name, orig_nu_o
        else:
            return img, label, case_name

    def __getitem__(self, index):
        return self.preprocessing(self.path_list[index])

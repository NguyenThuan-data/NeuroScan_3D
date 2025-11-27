import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BratsDataset(Dataset):
    def __init__(self, patient_ids, root_dir, mapping_df, target_shape=(128, 128, 128)):
        self.patient_ids = patient_ids
        self.root_dir = root_dir
        self.target_shape = target_shape
        # Map IDs to Grade (HGG=1, LGG=0)
        self.grade_map = {row['BraTS_2020_subject_ID']: 1 if row['Grade'] == 'HGG' else 0 
                          for _, row in mapping_df.iterrows()}

    # To get how many patients
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        path = os.path.join(self.root_dir, pid)
        
        # Load Images (Stack 4 modalities)
        images = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            p = os.path.join(path, f"{pid}_{mod}.nii")
            if not os.path.exists(p): p += ".gz"
            img = nib.load(p).get_fdata()
            # Normalize (Z-Score)
            mean, std = np.mean(img), np.std(img)
            if std > 0: img = (img - mean) / std
            images.append(img)
        img_stack = np.stack(images, axis=-1)
        
        # Load Mask -- seg
        p_seg = os.path.join(path, f"{pid}_seg.nii")
        if not os.path.exists(p_seg): p_seg += ".gz"
        if not os.path.exists(p_seg):
            for f in os.listdir(path):
                if 'seg' in f.lower():
                    p_seg = os.path.join(path, f)
                    break
        mask = nib.load(p_seg).get_fdata()
        mask[mask == 4] = 3 # Fix Label 4 -> 3
        
        # Crop
        img_crop, mask_crop = self.center_crop(img_stack, mask)
        
        # D. To Tensor
        img_t = torch.from_numpy(img_crop).float().permute(3, 0, 1, 2)
        mask_t = torch.from_numpy(mask_crop).long()
        grade_t = torch.tensor(self.grade_map.get(pid, 0), dtype=torch.float32).unsqueeze(0)
        
        return img_t, mask_t, grade_t

    def center_crop(self, img, mask):
        D, H, W, _ = img.shape
        tD, tH, tW = self.target_shape
        sd, sh, sw = max(0, (D-tD)//2), max(0, (H-tH)//2), max(0, (W-tW)//2)
        return img[sd:sd+tD, sh:sh+tH, sw:sw+tW, :], mask[sd:sd+tD, sh:sh+tH, sw:sw+tW]
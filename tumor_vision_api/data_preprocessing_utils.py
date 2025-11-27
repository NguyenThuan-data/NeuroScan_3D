import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BratsDataset(Dataset):
    def __init__(self, patient_ids, root_dir, mapping_df=None, target_shape=(128, 128, 128)):
        self.patient_ids = patient_ids
        self.root_dir = root_dir
        self.target_shape = target_shape
        
        # Map IDs to Grade (HGG=1, LGG=0)
        # Make mapping_df optional for inference
        if mapping_df is not None:
            self.grade_map = {row['BraTS_2020_subject_ID']: 1 if row['Grade'] == 'HGG' else 0 
                              for _, row in mapping_df.iterrows()}
        else:
            self.grade_map = {}  # Empty map for inference (returns 0 by default)

    # To get how many patients
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        path = os.path.join(self.root_dir, pid)
        
        # Load Images (Stack 4 modalities)
        images = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            # Try multiple filename patterns
            possible_paths = [
                os.path.join(path, f"{pid}_{mod}.nii"),
                os.path.join(path, f"{pid}_{mod}.nii.gz"),
                os.path.join(path, f"{mod}.nii"),
                os.path.join(path, f"{mod}.nii.gz"),
            ]
            
            # Find the first matching file
            p = None
            for possible_path in possible_paths:
                if os.path.exists(possible_path):
                    p = possible_path
                    break
            
            # If still not found, search directory for files containing modality name
            if p is None:
                # Special handling for t1ce vs t1
                search_pattern = mod
                if mod == 't1ce':
                    # Look for t1ce or tice
                    for f in os.listdir(path):
                        if ('t1ce' in f.lower() or 'tice' in f.lower()) and (f.endswith('.nii') or f.endswith('.nii.gz')):
                            p = os.path.join(path, f)
                            break
                elif mod == 't1':
                    # Look for t1 but NOT t1ce
                    for f in os.listdir(path):
                        if 't1' in f.lower() and 't1ce' not in f.lower() and 'tice' not in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz')):
                            p = os.path.join(path, f)
                            break
                else:
                    # For flair, t2
                    for f in os.listdir(path):
                        if mod in f.lower() and (f.endswith('.nii') or f.endswith('.nii.gz')):
                            p = os.path.join(path, f)
                            break
            
            if p is None:
                raise FileNotFoundError(f"Could not find {mod} modality in {path}. Available files: {os.listdir(path)}")
            
            print(f"Loading {mod}: {p}")
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
        
        # Handle case where mask doesn't exist (inference on new data)
        if os.path.exists(p_seg):
            mask = nib.load(p_seg).get_fdata()
            mask[mask == 4] = 3 # Fix Label 4 -> 3
        else:
            # Create dummy mask for inference
            mask = np.zeros(img_stack.shape[:3])
        
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
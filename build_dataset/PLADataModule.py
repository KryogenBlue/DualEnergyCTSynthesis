
import numpy as np
import torch
from torch.utils.data import DataLoader


from utils.utils import normalize

torch.use_deterministic_algorithms(True, warn_only=True)
import os

from torch.utils.data import IterableDataset
from pytorch_lightning import LightningDataModule

class NPYIterableDataset(IterableDataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for file in self.files:
            data = np.load(file, allow_pickle=True).item()
            image = data['data']
            label = data['label']
            patient_id = data['patient_id']
            image_id = data['image_id']
            yield torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), str(patient_id), str(image_id)

class NanjingPLA_DECT(LightningDataModule):
    def __init__(self,datatype,train_root_dir, valid_root_dir,test_root_dir, batch_size, gt_shape):
        super().__init__()
        self.datatype = datatype
        self.train_root_dir = train_root_dir
        self.valid_root_dir = valid_root_dir
        self.test_root_dir = test_root_dir
        self.batch_size = batch_size
        self.gt_shape = gt_shape
        self.train_mean,self.val_mean=0,0
        self.train_std,self.val_std=1,1

    def train_dataloader(self):
        train_dataset = NPYIterableDataset(self.train_root_dir)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=32,pin_memory=True,
        prefetch_factor=2)

    def val_dataloader(self):
        valid_dataset = NPYIterableDataset(self.valid_root_dir)
        return DataLoader(valid_dataset, batch_size=1, num_workers=16)

    def test_dataloader(self):
        test_dataset = NPYIterableDataset(self.test_root_dir)
        return DataLoader(test_dataset, batch_size=1, num_workers=4)

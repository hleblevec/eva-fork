import yaml, os, pdb
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import GLCDataset
from .transforms import get_transforms

class GLCDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.batch_size = config["batch_size"]
        self.test_splits_list = config["test_splits_list"]
        self.train_transforms = get_transforms(dvs_res=config["dvs_res"], augmentation=True)
        self.val_transforms = get_transforms(dvs_res=config["dvs_res"], augmentation=False)
        self.splits = yaml.safe_load(open(os.path.join(self.config["in_dir"], 'config.yaml'), 'r')) 
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        train_indices = self.splits.get('train_indices', [])
        val_indices = self.splits.get('val_indices', [])
        test_indices = self.splits.get('test_indices', [])
        test_dpu_indices = self.splits.get('test_dpu_indices', [])

        if (stage == 'fit' or stage is None) and (self.train_dataset is None or self.val_dataset is None):
            self.train_dataset = GLCDataset(indexes=train_indices, config=self.config, transforms=self.train_transforms)
            self.val_dataset = GLCDataset(indexes=val_indices, config=self.config, transforms=self.val_transforms)

        if stage == 'validate' or stage is None: 
            self.val_dataset = GLCDataset(indexes=val_indices, config=self.config, transforms=self.val_transforms)

        if stage == 'test' or stage is None:
            self.test_dataset = GLCDataset(indexes=test_indices, config=self.config, transforms=self.val_transforms) 
            self.test_dpu_indices = GLCDataset(indexes=test_dpu_indices, config=self.config, transforms=self.val_transforms) 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=0, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=0, batch_size=self.batch_size, pin_memory=True)

    def test_dataloader(self):
        list_tests_dataloader = []
        if 1 in self.test_splits_list:
            tdl_1 = DataLoader(self.test_dataset, num_workers=0, batch_size=self.batch_size, pin_memory=True)
            list_tests_dataloader.append(tdl_1)
        if 2 in self.test_splits_list:
            tdl_2 = DataLoader(self.test_dpu_indices, num_workers=0, batch_size=self.batch_size, pin_memory=True) 
            list_tests_dataloader.append(tdl_2)
        return list_tests_dataloader

import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from utils.dataset import CustomDataset

data_augmentation = {
    'train': A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5),
        A.RandomRotate90(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=int(256 * 0.2), max_width=int(256 * 0.2),
                        min_holes=1, min_height=int(256 * 0.1), min_width=int(256 * 0.1),
                        fill_value=0, p=0.5),
        ToTensorV2()], p=1.),

    'valid': A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
}

def train_data_loader(args):
    args = args
    augmentation = data_augmentation['train']
    df = pd.read_csv(args.train_df)

    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = True
    shuffle = True

    dataset = CustomDataset(df, augmentation)

    return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          shuffle=shuffle)

def valid_data_loader(args):
    args = args
    augmentation = data_augmentation['valid']
    df = pd.read_csv(args.valid_df)

    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = True
    shuffle = False

    dataset = CustomDataset(df, augmentation)

    return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          shuffle=shuffle)

# class train_data_loader(DataLoader):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.augmentation = data_augmentation['train']
#         self.df = args.train_df
#
#         self.batch_size = args.batch_size
#         self.num_workers = args.num_workers
#         self.pin_memory = True
#         self.shuffle = True
#
#         self.dataset = CustomDataset(self.df, self.augmentation)
#
#     def load(self):
#         return DataLoader(dataset=self.dataset,
#                           batch_size=self.batch_size,
#                           num_workers=self.num_workers,
#                           pin_memory=self.pin_memory,
#                           shuffle=self.shuffle)


# class valid_data_loader(DataLoader):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.augmentation = data_augmentation['valid']
#         self.df = args.valid_df
#
#         self.batch_size = args.batch_size
#         self.num_workers = args.num_workers
#         self.pin_memory = True
#         self.shuffle = True
#
#         self.dataset = CustomDataset(self.df, self.augmentation)
#
#     def load(self):
#         return DataLoader(dataset=self.dataset,
#                           batch_size=self.batch_size,
#                           num_workers=self.num_workers,
#                           pin_memory=self.pin_memory,
#                           shuffle=self.shuffle)

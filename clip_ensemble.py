import os
import math
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import cv2
from sklearn import preprocessing
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional
from zipfile import ZipFile
from pprint import pprint
import time
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.nn import LayerNorm
import torch.nn.functional as F
from collections import OrderedDict
import open_clip
from open_clip import get_pretrained_cfg
from tqdm import tqdm


class config:
    seed = 1997
    epochs = 100
    batch_size = 128

    clip_model_name1 = "ViT-H-14"
    clip_model_name2 = "ViT-L-14"
    clip_model_name3 = "ViT-B-32"
    clip_half = True
    embed_dim = 256
    neck_id = 0

    num_classes = 50
    # checkpoint_path = "./pretrained/fold0_epoch005_tloss18.2422.bin"

# checkpoint = torch.load(config.checkpoint_path)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()

class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m):
        super(AdaCos, self).__init__()
        self.num_features = in_features
        self.n_classes = out_features
        self.s = math.sqrt(2) * math.log(out_features - 1)
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output

class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine


def get_clip_model(model_name):
    clip_model = open_clip.create_model(model_name=model_name)

    if model_name == "ViT-H-14":
        pretrained_name = 'laion2b_s32b_b79k'
        pretrained_path = "./pretrained/vit-h-14.bin"
    elif model_name == "ViT-L-14-336":
        pretrained_name = 'openai'
        pretrained_path = './pretrained/vit-L-14-336.bin'
    elif model_name == "ViT-L-14":
        pretrained_name = 'laion2b_s32b_b82k'
        pretrained_path = './pretrained/vit-L-14.bin'
    elif model_name == "ViT-B-32":
        pretrained_name = 'laion2b_s34b_b79k'
        pretrained_path = './pretrained/vit-B-32.bin'

    clip_model.load_state_dict(torch.load(pretrained_path))
    clip_model = clip_model.cuda()

    # get mean/std
    if pretrained_name == 'openai':
        pretrained_cfg = {}
    else:
        pretrained_cfg = get_pretrained_cfg(model_name, pretrained_name)

    # set image / mean metadata from pretrained_cfg if available, or use default
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

    clip_model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
    clip_model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    return clip_model, pretrained_path


clip_model1, pretrained_path1 = get_clip_model(model_name=config.clip_model_name1)
clip_model2, pretrained_path2 = get_clip_model(model_name=config.clip_model_name2)
clip_model3, pretrained_path3 = get_clip_model(model_name=config.clip_model_name3)


class GUIEModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.clip_half = config.clip_half

        self.encoder1 = clip_model1.visual.eval()
        self.encoder_output_dim1 = clip_model1.visual.proj.shape[1]
        self.mean1 = clip_model1.visual.image_mean
        self.std1 = clip_model1.visual.image_std
        self.image_size1 = clip_model1.visual.image_size

        self.encoder2 = clip_model2.visual.eval()
        self.encoder_output_dim2 = clip_model2.visual.proj.shape[1]
        self.mean2 = clip_model2.visual.image_mean
        self.std2 = clip_model2.visual.image_std
        self.image_size2 = clip_model2.visual.image_size

        self.encoder3 = clip_model3.visual.eval()
        self.encoder_output_dim3 = clip_model3.visual.proj.shape[1]
        self.mean3 = clip_model3.visual.image_mean
        self.std3 = clip_model3.visual.image_std
        self.image_size3 = clip_model3.visual.image_size

        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False

        if config.neck_id == 0:
            self.neck = nn.Sequential(
                nn.Linear(self.encoder_output_dim1 + self.encoder_output_dim2 + self.encoder_output_dim3,
                          config.embed_dim),
                nn.BatchNorm1d(config.embed_dim),
                nn.PReLU()
            )
        elif config.neck_id == 1:
            self.neck = nn.Sequential(
                nn.Linear(self.encoder_output_dim1 + self.encoder_output_dim2 + self.encoder_output_dim3, 256),
                nn.BatchNorm1d(256),
                nn.PReLU(),
                nn.Linear(256, config.embed_dim),
                nn.BatchNorm1d(config.embed_dim),
                nn.PReLU()
            )
        elif config.neck_id == 2:
            self.neck = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.encoder_output_dim1 + self.encoder_output_dim2 + self.encoder_output_dim3,
                          config.embed_dim),
                nn.BatchNorm1d(config.embed_dim),
                nn.PReLU()
            )
        elif config.neck_id == 3:
            self.neck = nn.Sequential(
                nn.Linear(self.encoder_output_dim1 + self.encoder_output_dim2 + self.encoder_output_dim3, 384),
                nn.PReLU(),
                nn.Linear(384, config.embed_dim),
                nn.BatchNorm1d(config.embed_dim)
            )

        self.head = ArcMarginProduct_subcenter(config.embed_dim, config.num_classes)

    def preprocess_image1(self, x):
        x = transforms.functional.resize(x, size=self.image_size1)
        x = x / 255.0
        x = transforms.functional.normalize(x, mean=self.mean1, std=self.std1)
        return x

    def preprocess_image2(self, x):
        x = transforms.functional.resize(x, size=self.image_size2)
        x = x / 255.0
        x = transforms.functional.normalize(x, mean=self.mean2, std=self.std2)
        return x

    def preprocess_image3(self, x):
        x = transforms.functional.resize(x, size=self.image_size3)
        x = x / 255.0
        x = transforms.functional.normalize(x, mean=self.mean3, std=self.std3)
        return x

    def extract_feature(self, x):
        x1 = self.preprocess_image1(x)
        if self.clip_half:
            x1 = self.encoder1(x1.half()).to(torch.float32)
        else:
            x1 = self.encoder1(x1)

        x2 = self.preprocess_image2(x)
        if self.clip_half:
            x2 = self.encoder2(x2.half()).to(torch.float32)
        else:
            x2 = self.encoder1(x2)

        x3 = self.preprocess_image3(x)
        if self.clip_half:
            x3 = self.encoder3(x3.half()).to(torch.float32)
        else:
            x3 = self.encoder1(x3)

        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.neck(x)
        return x

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.head(x)
        return x

# base_model = GUIEModel()
# base_model = nn.DataParallel(_bas e_model).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
# base_model.load_state_dict(checkpoint)


class SubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = clip_model1.visual.eval()
        self.encoder_output_dim1 = clip_model1.visual.proj.shape[1]
        self.mean1 = clip_model1.visual.image_mean
        self.std1 = clip_model1.visual.image_std
        self.image_size1 = clip_model1.visual.image_size

        self.encoder2 = clip_model2.visual.eval()
        self.encoder_output_dim2 = clip_model2.visual.proj.shape[1]
        self.mean2 = clip_model2.visual.image_mean
        self.std2 = clip_model2.visual.image_std
        self.image_size2 = clip_model2.visual.image_size

        self.encoder3 = clip_model3.visual.eval()
        self.encoder_output_dim3 = clip_model3.visual.proj.shape[1]
        self.mean3 = clip_model3.visual.image_mean
        self.std3 = clip_model3.visual.image_std
        self.image_size3 = clip_model3.visual.image_size

        self.neck = GUIEModel().neck

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1),
            nn.ReLU(),
        )
        for param in self.classifier.parameters():
            param.requires_grad = True

        self.reducer = nn.AdaptiveAvgPool1d(64)

        self.arc = AdaCos(in_features=64, out_features=50, m=0.5)

    def preprocess_image1(self, x):
        x = transforms.functional.resize(x, size=self.image_size1)
        x = x / 255.0
        x = transforms.functional.normalize(x, mean=self.mean1, std=self.std1)
        return x

    def preprocess_image2(self, x):
        x = transforms.functional.resize(x, size=self.image_size2)
        x = x / 255.0
        x = transforms.functional.normalize(x, mean=self.mean2, std=self.std2)
        return x

    def preprocess_image3(self, x):
        x = transforms.functional.resize(x, size=self.image_size3)
        x = x / 255.0
        x = transforms.functional.normalize(x, mean=self.mean3, std=self.std3)
        return x

    def forward(self, x):
        x1 = self.preprocess_image1(x)
        x1 = self.encoder1(x1)

        x2 = self.preprocess_image2(x)
        x2 = self.encoder2(x2)

        x3 = self.preprocess_image3(x)
        x3 = self.encoder3(x3)

        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.neck(x)
        x = self.classifier(x)
        x = self.reducer(x)
        x = self.arc(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, df, augmentation=None):
        self.df = df
        self.df.artist = preprocessing.LabelEncoder().fit_transform(df.artist.values)
        self.augmentation = augmentation
        self.df.path = df.img_path.values
        self.df.labels = df.artist.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.labels[index]

        if self.augmentation is not None:
            image = self.augmentation(image=image)["image"]

        return image, label

data_augmentation = {
    'train': A.Compose([
        A.Resize(336, 336),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5),
        A.RandomRotate90(),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        #     max_pixel_value=255.0,
        #     p=1.0
        # ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=int(336 * 0.2), max_width=int(336 * 0.2),
                        min_holes=1, min_height=int(336 * 0.1), min_width=int(336 * 0.1),
                        fill_value=0, p=0.5),
        A.RandomSizedCrop([64, 64], 336, 336, w2h_ratio=1.0, interpolation=1, always_apply=False, p=0.5),
        ToTensorV2()], p=1.),

    'valid': A.Compose([
        A.Resize(336, 336),
        # A.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        #     max_pixel_value=255.0,
        #     p=1.0
        # ),
        ToTensorV2()], p=1.)
}

def train_data_loader():
    augmentation = data_augmentation['train']
    df = pd.read_csv('./train.csv')

    batch_size = config.batch_size
    num_workers = 20
    pin_memory = True
    shuffle = True

    dataset = CustomDataset(df, augmentation)

    return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          shuffle=shuffle,
                          drop_last=True)

def valid_data_loader():
    augmentation = data_augmentation['valid']
    df = pd.read_csv('./test.csv')

    batch_size = config.batch_size
    num_workers = 20
    pin_memory = True
    shuffle = False

    dataset = CustomDataset(df, augmentation)

    return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          shuffle=shuffle,
                          drop_last=False)

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss().to(device)

    # two gpus
    _model = SubModel().to(device)
    model = nn.DataParallel(_model).to(device)
    # model = SubModel().to(device)

    best_loss = 1997

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    train_loader = train_data_loader()

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        model.train()
        running_loss = []
        current_loss = 0

        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (img, label) in bar:

            optimizer.zero_grad()

            img, label = img.to(device), label.to(device).long()
            logits = model(img)

            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().cpu().numpy())
            current_loss = np.mean(running_loss)

        TIME = time.time() - start

        epochs = config.epochs
        print(f'epoch : {epoch}/{epochs}    time : {TIME:.0f}s/{TIME * (epochs - epoch - 1):.0f}s')
        print(f'TRAIN_loss : [{current_loss:.5f}]')

        if best_loss > current_loss:
            best_loss = current_loss

            torch.save(model.state_dict(), './saved/best_loss.pth')
            print('Best Loss Model Saved.')

        if scheduler is not None:
            scheduler.step()

        if epoch == config.epochs:
            test(model, device)


def test(model, device):
    # model.load_state_dict(torch.load('./saved/best_loss.pth'))

    valid_loader = valid_data_loader()

    model_preds = []
    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for step, (img, _) in bar:
            img = img.to(device)
            features = model(img)

            model_preds += features.argmax(1).detach().cpu().numpy().tolist()

    df = pd.read_csv('./train.csv')
    le = preprocessing.LabelEncoder()
    le.fit(df.artist.values)
    model_preds = le.inverse_transform(model_preds)

    submit = pd.read_csv('./sample_submission.csv')
    submit['artist'] = model_preds
    submit.to_csv('./submit.csv', index=False)

train()
# test(model=model, device=device)


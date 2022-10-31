import time

import pandas as pd
import numpy as np
import torch
import torchmetrics
from torch import nn, optim
from tqdm import tqdm

from utils.dataloader import train_data_loader, valid_data_loader
from utils.model import convnext_xlarge
from utils.loss import FocalLoss
from sklearn import preprocessing



def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = FocalLoss(50).to(device)

    # two gpus
    _model = convnext_xlarge(args).cuda()
    model = nn.DataParallel(_model).to(device)

    best_loss = 1997


    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    train_loader = train_data_loader(args)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss = []
        current_loss = 0

        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (img, label) in bar:

            optimizer.zero_grad()

            img, label = img.to(device), label.to(device).long()
            logits = model(img, label)

            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()

            running_loss.append(loss.detach().cpu().numpy())
            current_loss = np.mean(running_loss)

        TIME = time.time() - start

        epochs = args.epochs
        print(f'epoch : {epoch}/{epochs}    time : {TIME:.0f}s/{TIME * (epochs - epoch - 1):.0f}s')
        print(f'TRAIN_loss : [{current_loss:.5f}]')

        if best_loss > current_loss:
            best_loss = current_loss

            torch.save(model.state_dict(), './saved/convnext_best_loss.pth')
            print('Best Loss Model Saved.')

        if scheduler is not None:
            scheduler.step()

        if epoch == args.epochs:
            test(args, model, device)

        # if epoch == args.epochs:
        #     print(val_best_loss, val_best_acc)

def test(args, model, device):

        model.load_state_dict(torch.load('./saved/convnext_best_loss.pth'))

        valid_loader = valid_data_loader(args)

        model_preds = []

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
        submit.to_csv('./submit_convnext.csv', index=False)
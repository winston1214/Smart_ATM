import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision.transforms as T

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
from facial_model import Facial_model
from facial_dataloader import CustomDatasets
import argparse


def face_train(opt):
    BATCH_SIZE,EPOCH,lr = opt.batch,opt.epochs,opt.lr
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 디바이스 설정
    train_transform = T.Compose([
                    T.ToTensor(),
                    T.Resize((640,640)),
                    #T.Normalize([train_meanB,train_meanG,train_meanR],[train_stdB,train_stdG,train_stdR])
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = CustomDatasets(f'{opt.root}/',opt.csv,transform = train_transform)
    trainset,valset = D.random_split(dataset,[int(len(dataset)*0.8), int(len(dataset)*0.2)], generator=torch.Generator().manual_seed(42))
    '''
    ## Normalization
    train_meanRGB = [np.mean(x.numpy(),axis = (1,2)) for x,_ in trainset] # 채널별 평균
    train_stdRGB = [np.std(x.numpy(),axis = (1,2)) for x,_ in trainset] # 채널별 std

    train_meanR = np.mean([m[0] for m in train_meanRGB]) 
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])

    train_stdR = np.mean([m[0] for m in train_stdRGB])
    train_stdG = np.mean([m[1] for m in train_stdRGB])
    train_stdB = np.mean([m[2] for m in train_stdRGB])

    val_meanRGB = [np.mean(x.numpy(),axis=(1,2)) for x,_ in valset]
    val_stdRGB = [np.std(x.numpy(),axis=(1,2)) for x,_ in valset]

    val_meanR = np.mean([m[0] for m in val_meanRGB])
    val_meanG = np.mean([m[1] for m in val_meanRGB])
    val_meanB = np.mean([m[2] for m in val_meanRGB])

    val_stdR = np.mean([s[0] for s in val_stdRGB])
    val_stdG = np.mean([s[1] for s in val_stdRGB])
    val_stdB = np.mean([s[2] for s in val_stdRGB])
    '''

    #trainset.transform = train_transform
    #valset.transform = val_transform

    trainloader = D.DataLoader(trainset,batch_size = BATCH_SIZE,shuffle=True,drop_last=True)
    valloader = D.DataLoader(valset,batch_size = BATCH_SIZE,shuffle=True,drop_last=True)
    
    model = Facial_model()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 0.1,eta_min = 1e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    
    
    total_step = len(trainloader)
    best_val_acc = 0
    loss_list = []
    train_plt_list = []
    for e in tqdm(range(EPOCH)):
        train_acc_list = []
        running_loss = 0

        model.train()
        for i,(images,labels) in tqdm(enumerate(trainloader)):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.to(device=device, dtype=torch.int64)

            optimizer.zero_grad()

            probs = model(images)
            loss = criterion(probs,labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            # print(probs)
            preds = torch.argmax(probs,1)
            preds = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            

            
            batch_acc = (labels == preds).mean()
            train_acc_list.append(batch_acc)
        loss_list.append(running_loss/total_step)
        train_acc = np.mean(train_acc_list)
        train_plt_list.append(train_acc)
        print(f'Epochs : {e+1}/{EPOCH}, loss : {running_loss/total_step}, Train accuracy : {train_acc}')

        model.eval()
        val_acc_list = []
        val_plt_list = []
        with torch.no_grad():

            for images,labels in valloader:
                images = images.type(torch.FloatTensor).to(device)
                labels = labels.to(device=device, dtype=torch.int64)
                probs = model(images)
                loss = criterion(probs,labels)
                preds = torch.argmax(probs,1)
                preds = preds.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                batch_acc = (labels == preds).mean()
                val_acc_list.append(batch_acc)
            val_acc = np.mean(val_acc_list)
            val_plt_list.append(val_acc)
            print(f'validation acc : {val_acc}')
        lr_scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str,help = 'data root')
    parser.add_argument('--csv',type=str,help = 'label csv')
    parser.add_argument('--batch',type=int,default = 64,help = 'batch size')
    parser.add_argument('--epochs',type=int,default = 100,help = 'epoch')
    parser.add_argument('--lr',type=float,default = 0.001,help='learning_rate')
    opt = parser.parse_args()
    face_train(opt)
            
            
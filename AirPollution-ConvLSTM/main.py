#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST
from data.air_prediction import AirPrediction
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse

TIMESTAMP = "2020-03-09T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=100, type=int, help='sum of epochs')
parser.add_argument('--filepath',
                    default='data/mydata/pollutionPM25.h5',
                    type=str,
                    help='file path')
parser.add_argument('--savedir',
                    default='./save_model/',
                    type=str,
                    help='save dir')
parser.add_argument('--testing', action='store_true', help='testing')

args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = args.savedir #'./save_model/' #+ TIMESTAMP

trainData = AirPrediction(dataset_type=1,
                          filepath=args.filepath,
                          timesteps=12)
validData = AirPrediction(dataset_type=2,
                          filepath=args.filepath,
                          timesteps=12)
trainLoader = torch.utils.data.DataLoader(trainData,
                                          batch_size=args.batch_size,
                                          shuffle=True)
validLoader = torch.utils.data.DataLoader(validData,
                                          batch_size=args.batch_size,
                                          shuffle=False)

if args.convlstm:
    print('convlstm')
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
elif args.convgru:
    print('convgru')
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    print('convgru')
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def train():
    '''
    main function to run the training
    '''
    print('Training . . .')
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    #print(net)
    run_dir = './runs/' #+ TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (inputVar, targetVar, maskVar) in enumerate(t):
        #for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            label = targetVar.to(device)  # B,S,C,H,W
            mask = maskVar.to(device) # B,S,C,H,W
            
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            pred = pred * mask
            
            pred = trainData.scaler.inverse_transform(pred)
            label = trainData.scaler.inverse_transform(label)
                                                      
            #loss = lossfunction(pred, label) * trainData.loss_ratio
            loss = masked_mae_loss(pred, label)
            loss_aver = loss.item()
            #loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            #for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            for i, (inputVar, targetVar, maskVar) in enumerate(t):
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                mask = maskVar.to(device)
                
                pred = net(inputs)
                pred = pred * mask
                
                pred = validData.scaler.inverse_transform(pred)
                label = validData.scaler.inverse_transform(label)
                
                #loss = lossfunction(pred, label) * validData.loss_ratio
                loss = masked_mae_loss(pred, label)
                loss_aver = loss.item()
                #loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)

def test():
    '''
    main function to run the training
    '''
    print('Testing . . .')
    
    testData = AirPrediction(dataset_type=3,
                          filepath=args.filepath,
                          timesteps=12)
    
    testLoader = torch.utils.data.DataLoader(testData,
                                          batch_size=args.batch_size,
                                          shuffle=False)
    
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    run_dir = './runs/' #+ TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0

    ######################
    # test the model #
    ######################
    test_maes = []
    with torch.no_grad():
        net.eval()
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, (inputVar, targetVar, maskVar) in enumerate(t):
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            mask = maskVar.to(device)

            pred = net(inputs)
            pred = pred * mask

            pred = validData.scaler.inverse_transform(pred)
            label = validData.scaler.inverse_transform(label)
            #a = label.cpu().numpy()
            #b = pred.cpu().numpy()
            #print(sum(a[0].flatten()))
            #print(sum(b[0].flatten()))
            #reak

            mae_aver = masked_mae_loss(pred, label)
            #mae_aver = mae.item() / args.batch_size
            # record validation loss
            test_maes.append(mae_aver.item())

        torch.cuda.empty_cache()
        # calculate average loss
        test_mae = np.average(test_maes)
        print_msg = (f'test mae: {test_mae:.6f}')
        print(print_msg)


if __name__ == "__main__":
    if args.testing:
        test()
    else:
        train()

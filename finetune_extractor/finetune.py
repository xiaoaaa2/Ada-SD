import argparse
import os
import pandas as pd
import pdb
import random
import soundfile as sf
import time
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from itertools import chain

from collections import defaultdict
from datetime import date, datetime
from model import Cnceleb_sample_cssl, Cnceleb12_sample_cssl, Cnceleb_sample_cssl2
from model import Resnet101SpeakerCSSL
from pathlib import Path
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Config
from transformers.optimization import get_linear_schedule_with_warmup
from utils import EarlyStopping, BalancedBatchSampler
from models.resnet101 import *
from macls.data_utils.featurizer import AudioFeaturizer
from utils import AngProtoLoss4

parser = argparse.ArgumentParser(description="Fine-tuning ResNet101")
parser.add_argument('--base_path', type=str, 
                    default='/home/dataset/CN-Celeb_flac/data',
                    help='base location of the data')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for training')
parser.add_argument('--model_type', type=str, 
                    default='model/Resnet',
                    help='pretrained model name or path to trained model')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--pct_warmup_steps', type=float, default=0.1,
                    help='percentage of total steps for lr warmup')
parser.add_argument('--lr_scheduler', type=int, default=1,
                    help='with lr scheduler or without, 1=with, 0=constant')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs')
parser.add_argument('--num_freeze_steps', type=int, default=0,
                    help='number of epochs to freeze model layers')
parser.add_argument('--logfile', type=str, default='logs/log_1.log',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=100,
                    help='random_seed')
parser.add_argument('--optimizer_type', type=str, default="Adam",
                    help='type of optimizer: Adam or SGD')
parser.add_argument('--custom_embed_size', type=int, default=128,
                    help='add an extra FC layer and specify embed size/dim to be extracted')
parser.add_argument('--grad_acc_step', type=int, default=4,
                    help='number of steps to accumulate the gradient for')
parser.add_argument('--save_path', type=str, default="saved_models/test.pt",
                    help='path to save model')
parser.add_argument('--resume_training', type=int, default=0,
                    help='resume current training model, 1=resume, 0=load')
parser.add_argument('--data_type', type=str, default="ami",
                    help='To train with cn1 or cn1+2')
parser.add_argument('--with_relu', type=int, default=0,
                    help='with or without relu for FC layer and embedding')
parser.add_argument('--dropout_val', type=float, default=0,
                    help='additional dropout')
parser.add_argument('--refine_matrix', type=int, default=0,
                    help='whether to have thresholding in CE step of AP loss')
parser.add_argument('--num_classes_in_batch', type=int, default=0,
                    help='to specify a specific number of speakers in each batch, 0 for random number of speakers')
parser.add_argument('--g_blur', type=float, default=1.,
                    help='gaussian blur to use with contrastive loss. 0=no blur & use abs threshold')
parser.add_argument('--p_pct', type=int, default=100,
                    help='threshold for the AP and MSE loss')                 
parser.add_argument('--mse_fac', type=float, default=0.0,
                    help='weight of the mse loss')
parser.add_argument('--margin', type=float, default=0.0,
                    help='P percentile margin')

args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


logging("\n-------------------")
logging(f"date: {date.today()}")
logging(f"time: {datetime.now().strftime('%H:%M:%S')}")


def collate_pad_batch(batch):
    """
    pads data of different length to the same max length
    """
    seg_1 = [t[0] for t in batch]
    labels = torch.tensor([t[1] for t in batch])
    seg_1 = pad_sequence(seg_1, batch_first=True)
    return seg_1, labels


def eval(model, val_loader):
    model.to(device)
    model.eval()

    eval_loss = 0
    num_samples = 0

    with torch.no_grad(): 
        for seg_2, label in val_loader:
            seg_2 = seg_2.to(device)
            label = label.to(device)
            seg_2 = torch.transpose(seg_2, 1, 2)
            f2 = model(seg_2)
            loss = loss_fn(f2, label)
            eval_loss += float(loss.item()) * seg_2.size(0)
            num_samples += seg_2.size(0)
  
    logging(f"eval loss: {eval_loss/ num_samples}")

    return eval_loss


def train(train_loader):

    model = eval('ResNet101')(feat_dim=64, embed_dim=256)
    model.to(device)
    # Specify the path to your trained weight file
    model_weights_path = '/home/AISHELL-4/sd/VBx/models/ResNet101_16kHz/nnet/raw_81.pth'
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
  
    for param in model.parameters():
        param.requires_grad = False

    # 解冻倒数第一层
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
        
    for param in model.embedding.parameters():
        param.requires_grad = True
        
    for param in loss_fn.parameters():
        param.requires_grad = True
    arams_to_optimize = chain(filter(lambda p: p.requires_grad, model.parameters()),
                           filter(lambda p: p.requires_grad, loss_fn.parameters()))

    # Define optimizer  
    if args.optimizer_type == "Adam":
        optimizer = Adam(arams_to_optimize, lr=config["lr"])
    elif args.optimizer_type == "SGD":
        optimizer = SGD(arams_to_optimize, lr=config["lr"])
    else:
        raise ValueError('Optimizer must be one of Adam or SGD')
    
    t_total = len(train_loader) // config["grad_acc_step"] * config["epochs"]
    num_warmup_steps = int(t_total * config["pct_warmup_steps"])
    logging(f"total steps: {t_total}")
    if config["lr_scheduler"]:
        lr_scheduler = get_linear_schedule_with_warmup(
                            optimizer, 
                            num_warmup_steps, 
                            t_total)

    logging("Training model...")

    min_val_loss = float('inf')
    estop = EarlyStopping(best_loss=min_val_loss, patience=1, min_delta=0)

    num_steps = 0

    for epoch in range(config["epochs"]):

        logging(f"epoch {epoch}")

        start = time.time()

        running_loss = 0.0
        samples = 0
        curr_epoch_steps = 0
        
        model.train()
        # print(dataloader)
        # torch.cuda.empty_cache()
       
        for i, (seg_2, label) in enumerate(dataloader):
        
            # torch.cuda.set_per_process_memory_fraction(0.5)  # 可以根据需要调整分数
            # Manually free up GPU memory  
            
            seg_2 = seg_2.to(device)
            label = label.to(device)
                        
            if curr_epoch_steps == 800:
                torch.save(model.state_dict(), 'saved_models/model_800_3s.pt')
            
            # else:
            #     model.unfreeze_base()
            seg_2 = torch.transpose(seg_2, 1, 2)
            f2 = model(seg_2)
            # f2 = f2.data.cpu().numpy()
            # f2 = f2.detach()
            loss = loss_fn(f2, label)
            # loss, _ = model(seg_2, label)
            
            loss = loss / config["grad_acc_step"]
            # backward
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # grad accumulation
            if (i+1) % config["grad_acc_step"]  == 0:                
                optimizer.step()
                optimizer.zero_grad()
                if config["lr_scheduler"]:
                    lr_scheduler.step()
                num_steps += 1
                curr_epoch_steps += 1

            running_loss += float(loss.item()) * seg_2.size(0) * config['grad_acc_step']
            samples += seg_2.size(0)

            if curr_epoch_steps > 0 and curr_epoch_steps % 50 == 0 and (i+1) % curr_epoch_steps == 0:
                logging(f"epoch: {epoch} | new step: {curr_epoch_steps} | loss: {running_loss/samples} | time taken: {time.time() - start}")
           
        print("eval ---")

        # evaluate model
        eval_loss = eval_l(model, val_dataloader)

        logging(f"End of epoch {epoch} | time taken: {time.time() - start}")

        # save best model
        # but continue training till end
        estop(eval_loss)
        if eval_loss <= estop.best_loss:
            logging("Saving model...")
            torch.save(model.state_dict(), config["save_path"])
        else:
            logging("loss up")
            if config['epochs'] == 1:
                logging("Saving model coz only one epoch...")
                torch.save(model.state_dict(), config["save_path"])
    
    logging("Finished training!")



#### LOADING DATA #############
base_path = '/home/Pytorch/dataset/CN-Celeb_flac/data'
dev_path = '/home/Pytorch/dataset/CN-Celeb_flac/data'
dev_path2 = '/home/Pytorch/dataset/CN-Celeb2_flac/data'
train_path = '/home/Pytorch/dataset/CN-Celeb_flac/data'

if args.data_type == "vox_12":
    logging(f"using positive samples from other utterance Vox 12")
    train_file1 = '/home/mifs/epcl2/project/w2v2_sv/data/voxceleb1_train.csv'
    train_file2 = '/home/mifs/epcl2/project/w2v2_sv/data/voxceleb2_train.csv'
    val_file1 = '/home/mifs/epcl2/project/w2v2_sv/data/voxceleb1_val.csv'
    val_file2 = '/home/mifs/epcl2/project/w2v2_sv/data/voxceleb2_val.csv'
    
    train_df1 = pd.read_csv(train_file1)
    train_df2 = pd.read_csv(train_file2)
    val_df1 = pd.read_csv(val_file1)
    val_df2 = pd.read_csv(val_file2)
    
    train_data1 = Voxceleb12_sample_cssl2(dev_path, dev_path2, train_df1, train_df2)
    val_data1 = Voxceleb12_sample_cssl2(dev_path, dev_path2, val_df1, val_df2)

    train_df1 = pd.concat([train_df1, train_df2]).reset_index(drop=True)
    val_df1 = pd.concat([val_df1, val_df2]).reset_index(drop=True)

    label_column1 = 'speaker_id'
    logging(f"num classes: {train_df1[label_column1].nunique()}")

elif args.data_type == "cn_1":
    logging(f"using positive samples from other utterance cn 1")
    train_file = '/home/zidonghua/zcy/scale/cnceleb1_train.csv' 
    val_file = '/home/zidonghua/zcy/scale/cnceleb1_val.csv'    
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    train_data = Cnceleb_sample_cssl(train_path, train_df)
    val_data = Cnceleb_sample_cssl(dev_path, val_df)

    # train_df =[train_df]
    # val_df = [val_df]

    label_column = 'speaker_id'
    logging(f"num classes: {train_df[label_column].nunique()}")
    

elif args.data_type == "cn_2":
    logging(f"using positive samples from other utterance cn 1")
    train_file = '/home/zidonghua/zcy/scale/cnceleb2_train.csv' 
    val_file = '/home/zidonghua/zcy/scale/cnceleb2_val.csv'    
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    train_data = Cnceleb_sample_cssl(train_path, train_df)
    val_data = Cnceleb_sample_cssl(dev_path, val_df)

    # train_df =[train_df]
    # val_df = [val_df]

    label_column = 'speaker_id'
    logging(f"num classes: {train_df[label_column].nunique()}")
    
    
elif args.data_type == "cn_1_ft":
    logging(f"using positive samples from other utterance cn 1")
    train_file = '/home/zidonghua/zcy/scale/cnceleb1_train.csv' 
    val_file = '/home/zidonghua/zcy/scale/cnceleb1_val.csv'    
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    train_data = Cnceleb1ft_sample_cssl(train_path, train_df)
    val_data = Cnceleb1ft_sample_cssl(dev_path, val_df)

    # train_df =[train_df]
    # val_df = [val_df]

    label_column = 'speaker_id'
    logging(f"num classes: {train_df[label_column].nunique()}")
    
elif args.data_type == "cn_12":
    logging(f"using positive samples from other utterance Vox 12")
    train_file1 = '/home/zidonghua/zcy/scale/cnceleb1_train.csv'
    train_file2 = '/home/zidonghua/zcy/scale/cnceleb2_train.csv'
    val_file1 = '/home/zidonghua/zcy/scale/cnceleb1_val.csv'
    val_file2 = '/home/zidonghua/zcy/scale/cnceleb2_val.csv'
    
    train_df1 = pd.read_csv(train_file1)
    train_df2 = pd.read_csv(train_file2)
    val_df1 = pd.read_csv(val_file1)
    val_df2 = pd.read_csv(val_file2)
    
    train_data = Cnceleb12_sample_cssl(dev_path, dev_path2, train_df1, train_df2)
    val_data = Cnceleb12_sample_cssl(dev_path, dev_path2, val_df1, val_df2)

    train_df = pd.concat([train_df1, train_df2]).reset_index(drop=True)
    val_df = pd.concat([val_df1, val_df2]).reset_index(drop=True)

    label_column = 'speaker_id'
    logging(f"num classes: {train_df[label_column].nunique()}")
   

# ensures that there are num_classes_in_batch diff speakers in each batch
if args.num_classes_in_batch:
    train_batch_sampler = BalancedBatchSampler(data_df=train_df,
                            n_classes=args.num_classes_in_batch,
                            n_samples=int(args.batch_size//args.num_classes_in_batch),
                            label_column_name=label_column)
    dataloader = DataLoader(
        train_data,
        batch_sampler=train_batch_sampler, 
        collate_fn=collate_pad_batch,
        num_workers=0
    )

    val_batch_sampler = BalancedBatchSampler(data_df=val_df,
                            n_classes=args.num_classes_in_batch,
                            n_samples=int(args.batch_size//args.num_classes_in_batch),
                            label_column_name=label_column)
    val_dataloader = DataLoader(
        val_data,
        batch_sampler=val_batch_sampler, 
        collate_fn=collate_pad_batch,
        num_workers=0
    )
else:
    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pad_batch,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pad_batch,
        num_workers=0
    )


if args.dropout_val:
    w2v2_config = Wav2Vec2Config(hidden_dropout = args.dropout_val,
            activation_dropout = args.dropout_val,
            attention_dropout = args.dropout_val,
            feat_proj_dropout = args.dropout_val,
            feat_quantizer_dropout = args.dropout_val,
            final_dropout = args.dropout_val,
            layerdrop = args.dropout_val)
else:
    w2v2_config = Wav2Vec2Config()


config = { 
    "base_path": args.base_path,
    "batch_size": args.batch_size,
    "num_classes_in_batch": args.num_classes_in_batch,
    "w2v2_model": args.model_type,
    "device": device,
    "lr": args.lr,
    "pct_warmup_steps": args.pct_warmup_steps,
    "num_freeze_steps": args.num_freeze_steps,
    "epochs": args.epochs,
    "optimizer_type": args.optimizer_type,
    "grad_acc_step": args.grad_acc_step,
    "save_path": args.save_path,
    "resume_training": args.resume_training,
    "lr_scheduler":args.lr_scheduler,
    "data_type": args.data_type,
    "custom_embed_size": args.custom_embed_size,
    "with_relu": args.with_relu,
    "dropout_val": args.dropout_val,
    "refine_matrix": args.refine_matrix,
    "g_blur": args.g_blur,
    "p_pct": args.p_pct,
    "mse_fac": args.mse_fac,
    "margin": args.margin,
    "w2v2_config": w2v2_config,
}


loss_fn = AngProtoLoss4(
            config,
            device=config["device"], 
            refine_matrix=config['refine_matrix'], 
            g_blur=config['g_blur'],
            p_pct=config['p_pct'],
            mse_fac=config['mse_fac'])


for item in config:
    logging(f"{item}: {config[item]}")
# print('dataloader', dataloader)
train(dataloader)



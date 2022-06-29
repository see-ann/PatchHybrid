import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import nets.bagnet
import nets.resnet
from utils.defense_utils import *
from scipy.stats import iqr

import os 
import joblib
import argparse
from tqdm import tqdm
import numpy as np 
from scipy.special import softmax
from math import ceil
import PIL
from scipy import stats


parser = argparse.ArgumentParser()

parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--data_dir', default='data', type=str,help="path to data")
parser.add_argument('--dataset', default='imagenette', choices=('imagenette', 'imagenette_patch', 'imagenet','cifar'),type=str,help="dataset")
parser.add_argument("--model",default='bagnet17',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='none',type=str,help="aggregation methods. set to none for local feature")
parser.add_argument("--skip",default=1,type=int,help="number of example to skip")
parser.add_argument("--thres",default=0.0,type=float,help="detection threshold for robust masking")
parser.add_argument("--patch_size",default=-1,type=int,help="size of the adversarial patch")
parser.add_argument("--m",action='store_true',help="use robust masking")
parser.add_argument("--cbn",action='store_true',help="use cbn")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)
DATASET = args.dataset
def get_dataset(ds,data_dir):
    
    # adversarial images that we created already underwent the Resize
    # CenterCrop, and ToTensor functions, so only need to normalize here.
    if ds == 'imagenette_patch':
       ds_dir=os.path.join(data_dir,'val')
       ds_transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       dataset_ = datasets.ImageFolder(ds_dir,ds_transforms)
       class_names = dataset_.classes

    if ds in ['imagenette','imagenet']:
        ds_dir=os.path.join(data_dir,'val')
        ds_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        dataset_ = datasets.ImageFolder(ds_dir,ds_transforms)
        class_names = dataset_.classes
    elif ds == 'cifar':
        ds_transforms = transforms.Compose([
            transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_ = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=ds_transforms)
        class_names = dataset_.classes
    return dataset_,class_names

val_dataset_,class_names = get_dataset(DATASET,DATA_DIR)
skips = list(range(0, len(val_dataset_), args.skip))
val_dataset = torch.utils.data.Subset(val_dataset_, skips)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,shuffle=False)

#build and initialize model
device = 'cuda' #if torch.cuda.is_available() else 'cpu'

if args.clip > 0:
    clip_range = [0,args.clip]
else:
    clip_range = None

if 'bagnet17' in args.model:
    model = nets.bagnet.bagnet17(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=17
elif 'bagnet33' in args.model:
    model = nets.bagnet.bagnet33(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=33
elif 'bagnet9' in args.model:
    model = nets.bagnet.bagnet9(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=9


if DATASET == 'imagenette' or DATASET == 'imagenette_patch':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_nette.pth'))
    model.load_state_dict(checkpoint['model_state_dict']) 
    args.patch_size = args.patch_size if args.patch_size>0 else 32     
elif  DATASET == 'imagenet' :
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_net.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    args.patch_size = args.patch_size if args.patch_size>0 else 32 
elif  DATASET == 'cifar':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_192_cifar.pth'))
    model.load_state_dict(checkpoint['net'])
    args.patch_size = args.patch_size if args.patch_size>0 else 30


rf_stride=8
window_size = ceil((args.patch_size + rf_size -1) / rf_stride)
print("window_size",window_size)

    
model = model.to(device)
model.eval()
cudnn.benchmark = True

accuracy_list=[]
result_list=[]
clean_corr=0

skewness_list = []

counter = 0
for data,labels in tqdm(val_loader):

   
    counter += 1

    if counter == 2:
        break
    
    data=data.to(device)
    labels = labels.numpy()

    print("Labels: ")
    print(labels)
    print(len(labels))
    output_clean = model(data).detach().cpu().numpy() # logits
    #output_clean = softmax(output_clean,axis=-1) # confidence
    #output_clean = (output_clean > 0.2).astype(float) # predictions with confidence threshold

    #note: the provable analysis of robust masking is cpu-intensive and can take some time to finish
    #you can dump the local feature and do the provable analysis with another script so that GPU mempry is not always occupied  
    
    for i in range(len(labels)):
        if args.m:#robust masking
            local_feature = output_clean[i]
            result = provable_masking(local_feature,labels[i],thres=args.thres,window_shape=[window_size,window_size])
            result_list.append(result)
            clean_pred = masking_defense(local_feature,thres=args.thres,window_shape=[window_size,window_size])
            clean_corr += clean_pred == labels[i]
            
        elif args.cbn:#cbn 
            # note that cbn results reported in the paper is obtained with vanilla BagNet (without provable adversrial training), since
            # the provable adversarial training is proposed in our paper. We will find that our training technique also benifits CBN
            result = provable_clipping(output_clean[i],labels[i],window_shape=[window_size,window_size])
            result_list.append(result)
            clean_pred = clipping_defense(output_clean[i])
            clean_corr += clean_pred == labels[i]   
    print(f"clean pred: {clean_pred}")
    print(f"result: {result}")
    acc_clean = np.sum(np.argmax(np.mean(output_clean,axis=(1,2)),axis=1) == labels)
    accuracy_list.append(acc_clean)

    output_shape = output_clean.shape
    print(output_shape)
    logit_mgtds = np.linalg.norm(output_clean.reshape((output_shape[1]*output_shape[2], 10)), axis=1)



    skewness = (np.mean(logit_mgtds) - np.median(logit_mgtds))/np.std(logit_mgtds)
    

    skewness_list.append(skewness)


    output_shape = output_clean.shape
    print(f"output_clean shape: {output_shape}")
    logits_2d = output_clean.reshape(output_shape[1]*output_shape[2], 10)
    logit_mgtds = np.linalg.norm(logits_2d, axis=1)
    print(f"logits_2d shape: {logits_2d.shape}")
    print(f"logits_mgtds shape: {logit_mgtds.shape}")



#     skewness = (np.mean(logit_mgtds) - np.median(logit_mgtds))/np.std(logit_mgtds)
#     std = np.std(logit_mgtds)

#     skewness_list.append(skewness)
#     std_list.append(std)



# print("skewness_list: ")
# print(skewness_list)
# print(np.mean(skewness_list))
# print(np.mean(std_list))

for i in range(logits_2d.shape[1]):
    std = np.round(np.std(logits_2d[:, i]), decimals=2)
    print(f"std of class {i} evidence: {std}")
    fig, ax = plt.subplots(1, 1)
    ax.hist(logits_2d[:, i], bins = 40)
    ax.set_xlabel(f"Class {i} Evidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Local Class {i} Evidence")
    if 'bagnet17' in args.model:
        plt.savefig(f"class{i}_dist_bn17")

    elif 'bagnet33' in args.model:
        plt.savefig(f"class{i}_dist_bn33")

    elif'bagnet9' in args.model:
        plt.savefig(f"class{i}_dist_bn17")



fig, ax = plt.subplots(1, 1)
ax.hist(logit_mgtds, bins = 40)
ax.set_xlabel("Logit Magnitude")
ax.set_ylabel("Count")
ax.set_title("Distribution of Local Logit Magnitudes")

if 'bagnet17' in args.model:
    plt.savefig("logits_dist_bn17")

elif 'bagnet33' in args.model:
    plt.savefig("logits_dist_bn33")

elif 'bagnet9' in args.model:
    plt.savefig("logits_dist_bn9")

# cases,cnt=np.unique(result_list,return_counts=True)
# print("Provable robust accuracy:",cnt[-1]/len(result_list) if len(cnt)==3 else 0)
# print("Clean accuracy with defense:",clean_corr/len(result_list))
# print("Clean accuracy without defense:",np.sum(accuracy_list)/len(val_dataset))
# print("------------------------------")
# print("Provable analysis cases (0: incorrect prediction; 1: vulnerable; 2: provably robust):",cases)
# print("Provable analysis breakdown",cnt/len(result_list))
# print("------------------------------")
print(output_clean.shape)
print(logit_mgtds.shape)


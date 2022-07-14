from requests import patch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  torchvision import datasets, transforms

from torchvision.utils import save_image

import nets.bagnet
import nets.resnet
from utils.defense_utils import *

import os 
import argparse
from tqdm import tqdm
import numpy as np 
import PIL
from PatchAttacker import PatchAttacker
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--dump_dir",default='patch_adv',type=str,help="directory to save attack results")
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--data_dir', default='data', type=str,help="path to data")
parser.add_argument('--dataset', default='imagenette', choices=('imagenette','imagenet','cifar'),type=str,help="dataset")
parser.add_argument("--model",default='bagnet17',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. set to none for local feature")
parser.add_argument("--patch_size",type=int,help="size of the adversarial patch")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)
DATASET = args.dataset
DUMP_DIR=os.path.join('dump',args.dump_dir+'_{}_{}'.format(args.model,args.dataset))
if not os.path.exists('dump'):
    os.mkdir('dump')
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)



if DATASET in ['imagenette','imagenet']:
    DATA_DIR=os.path.join(DATA_DIR,'val')
    mean_vec = [0.485, 0.456, 0.406]
    std_vec =  [0.229, 0.224, 0.225]
    ds_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean_vec,std_vec)
        ])
    val_dataset = datasets.ImageFolder(DATA_DIR,ds_transforms)
    class_names = val_dataset.classes
elif DATASET == 'cifar':
    mean_vec = [0.4914, 0.4822, 0.4465]
    std_vec = [0.2023, 0.1994, 0.2010]
    ds_transforms = transforms.Compose([
        transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean_vec,std_vec),
    ])
    val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=ds_transforms)
    class_names = val_dataset.classes

# set batch_size = 1 for single images, shuffle=True for a variety of classes
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,shuffle=True)

#build and initialize model
device = 'cuda' #if torch.cuda.is_available() else 'cpu'

if args.clip > 0:
    clip_range = [0,args.clip]
else:
    clip_range = None

if 'bagnet17' in args.model:
    model = nets.bagnet.bagnet17(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'bagnet33' in args.model:
    model = nets.bagnet.bagnet33(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'bagnet9' in args.model:
    model = nets.bagnet.bagnet9(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'resnet50' in args.model:
    model = nets.resnet.resnet50(pretrained=True,clip_range=clip_range,aggregation=args.aggr)


if DATASET == 'imagenette':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_nette.pth'))
    model.load_state_dict(checkpoint['model_state_dict']) 
elif  DATASET == 'imagenet':
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_net.pth'))
    model.load_state_dict(checkpoint['state_dict'])
elif  DATASET == 'cifar':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_192_cifar.pth'))
    model.load_state_dict(checkpoint['net'])
    
model = model.to(device)
model.eval()
cudnn.benchmark = True


attacker = PatchAttacker(model, mean_vec, std_vec,patch_size=args.patch_size,step_size=0.05,steps=500)

adv_list=[]
error_list=[]
accuracy_list=[]
patch_loc_list=[]
 
# For filepaths of saved images
class_count = np.zeros(10, dtype = int)

for counter, (data,labels) in enumerate(tqdm(val_loader)):

    if counter == 100 :
        break

    data,labels=data.to(device),labels.to(device)
    
    # clean image
    data_clean = data
    
    # make the adversarial image
    data_adv,patch_loc = attacker.perturb(data, labels)

    # finally correct inverse tranform
    # needed to apply mean and std transforms separately
    ds_inverse_transforms = transforms.Compose([
       transforms.Normalize(mean = [ 0., 0., 0. ],
                            std = [1/x for x in std_vec]),
       transforms.Normalize(mean = [-x for x in mean_vec],
                           std = [ 1., 1., 1. ]),
       ])
    data_adv_copy = ds_inverse_transforms(data_adv)
    data_clean_copy = ds_inverse_transforms(data_clean)

    # Formatted filename
    label = int(labels[0])
    formatted_fn = f"class{label}_img{class_count[label]}.png"
    print(f"formatted filename: {formatted_fn}")
    
    # Save patch and clean version of image
    if 'bagnet17' in args.model:
        mod = 'bn17'
    elif 'bagnet33' in args.model:
        mod = 'bn33'
    elif 'bagnet9' in args.model:
        mod = 'bn9'
    elif 'resnet50' in args.model:
        mod = 'rn50'

    save_path = f"./data/imagenette_pair_{mod}/val/{label}/{formatted_fn}"

    if not os.path.exists(save_path):
       os.makedirs(save_path)

    class_count[label] += 1

    output_adv = model(data_adv)

    print(f"True label: {label}")
    print(f"Model output: {torch.argmax(output_adv, dim=1).cpu().detach().numpy()[0]}")

    error_adv=torch.sum(torch.argmax(output_adv, dim=1) != labels).cpu().detach().numpy()
    output_clean = model(data)
    acc_clean=torch.sum(torch.argmax(output_clean, dim=1) == labels).cpu().detach().numpy()

    data_adv=data_adv.cpu().detach().numpy()
    patch_loc=patch_loc.cpu().detach().numpy()
    
    if (error_adv) :
        print("Successful attack!")

        # Save successful attacks to test RF hypothesis
        save_image(data_adv_copy, os.path.join(save_path, f"SUCC_patch_{formatted_fn}"))
        save_image(data_clean_copy, os.path.join(save_path, f"SUCC_clean_{formatted_fn}"))

        print(f"Saved successful attack image {formatted_fn}")
    
    else :
        print ("Unsuccessful attack :(")

    patch_loc_list.append(patch_loc)
    adv_list.append(data_adv)
    error_list.append(error_adv)
    accuracy_list.append(acc_clean)
    print("\n")


adv_list = np.concatenate(adv_list)
error_arr = np.array(error_list)
patch_loc_list = np.concatenate(patch_loc_list)
joblib.dump(adv_list,os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size)))
joblib.dump(patch_loc_list,os.path.join(DUMP_DIR,'patch_loc_list_{}.z'.format(args.patch_size)))
#print("Attack success rate:",np.sum(error_list)/len(val_dataset))
#print("Clean accuracy:",np.sum(accuracy_list)/len(val_dataset))
print("Attack success rate:",np.sum(error_list)/counter)
print("Clean accuracy:",np.sum(accuracy_list)/counter)
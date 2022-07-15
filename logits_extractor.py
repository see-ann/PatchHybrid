##############################################################################################################
# Part of code adapted from https://github.com/alevine0/patchSmoothing/blob/master/certify_imagenet_band.py
##############################################################################################################

from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import nets.dsresnet_imgnt as resnet_imgnt
import nets.dsresnet_cifar as resnet_cifar
from torchvision import datasets,transforms
from tqdm import tqdm
from utils.defense_utils import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import os
import nets.resnet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--patch_size', default=-1, type=int, help='patch_size')
parser.add_argument('--dataset', default='imagenette', choices=('imagenette', 'imagenette_pair_rn50'),type=str,help="dataset")
parser.add_argument('--data_dir', default='data', type=str,help="path to data")

parser.add_argument('--skip', default=1,type=int, help='Number of images to skip')
args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)
DATASET = args.dataset

device = 'cuda' #if torch.cuda.is_available() else 'cpu'

cudnn.benchmark = True

def get_dataset(ds,data_dir):
    if ds in ['imagenette', 'imagenette_pair_rn50']:
        ds_dir=os.path.join(data_dir,'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset_ = datasets.ImageFolder(ds_dir, transforms.Compose([
                transforms.Resize((299,299)), #note that here input size if 299x299 instead of 224x224
                transforms.ToTensor(),
                normalize,
            ]))
    return dataset_,dataset_.classes

val_dataset_,class_names = get_dataset(DATASET,DATA_DIR)
skips = list(range(0, len(val_dataset_), args.skip))
val_dataset = torch.utils.data.Subset(val_dataset_, skips)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,shuffle=False)

num_cls = len(class_names)

# Model
print('==> Building model..')



if DATASET == 'imagenette' or DATASET == 'imagenette_pair_rn50':
    net = resnet_imgnt.resnet50()
    net = torch.nn.DataParallel(net)
    num_ftrs = net.module.fc.in_features
    net.module.fc = nn.Linear(num_ftrs, num_cls)  
    checkpoint = torch.load(os.path.join(MODEL_DIR,'ds_nette.pth'))
    args.patch_size = args.patch_size if args.patch_size>0 else 42

model = nets.resnet.resnet50(pretrained=True,clip_range=None,aggregation=None)
print(f"Patch Size: {args.patch_size}")


net.load_state_dict(checkpoint['net'])

net = net.to(device)
net.eval()


correct = 0
cert_correct = 0
cert_incorrect = 0
total = 0
counter = 0
with torch.no_grad():
    for inputs, targets in tqdm(val_loader):
        if counter == 16:
            break
        sample_fname = val_loader.sampler.data_source.dataset.imgs[counter][0]
        sample_fname_list = sample_fname.split('/')
        file_name = sample_fname_list[-1]
        folder_name = sample_fname_list[-2]
        label_name = sample_fname_list[-3]
        print(f"file name: {file_name}")
        print(f"folder name: {folder_name}")
        print(f"label name: {label_name}")




        inputs, targets = inputs.to(device), targets.to(device)

        output_clean = model(inputs).detach().cpu().numpy() # logits
        print("OUTPUT CLEAN: ")
        print(output_clean)
        
#         total += targets.size(0)
#         # predictions,  certyn, logits_2d  = ds(inputs, net,args.band_size, args.patch_size, num_cls,threshold = 0.2)
        
        
        
#         correct += (predictions.eq(targets)).sum().item()
#         cert_correct += (predictions.eq(targets) & certyn).sum().item()
#         cert_incorrect += (~predictions.eq(targets) & certyn).sum().item()

#         print("Predictions: ")
#         print(predictions.cpu().numpy())


#         print("Logits: ")
#         print(logits_2d)
#         print(logits_2d.shape)

#         logit_mgtds = np.linalg.norm(logits_2d, axis=1)
        
        

#         save_path = f"./plots/rn50_patch_plots/ds_{args.band_size}/{label_name}/{folder_name}"

#         save_path_names = [save_path+"/class_evidence", save_path+"/logit_mgtds_hist", save_path+"/logit_mgtds_box"]

#         for path in save_path_names:
#             if not os.path.exists(path):
#                 os.makedirs(path)

#         if counter == 0:
#             f= open(f"{save_path}/info_clean.txt","w+")
#         if counter == 1:
#             f= open(f"{save_path}/info_adv.txt","w+")


#         max_class = None
#         max_sum = -float("inf")
#             # Class evidence histograms
#         for i in range(logits_2d.shape[1]):
#             sum = np.sum(logits_2d[:, i])
#             sum = np.floor(sum)
#             print(f"sum of class {i} evidence: {sum}")

#             if sum > max_sum:
#                 max_class = i
#                 max_sum = sum
#             fig, ax = plt.subplots(1, 1)
#             ax.hist(logits_2d[:, i], bins = 40)
#             ax.set_xlabel(f"Class {i} Evidence")
#             ax.set_ylabel("Count")
#             ax.set_title(f"Distribution of Local Class {i} Evidence")

#             if counter % 2 == 0: # even counters are clean
#                 plt.savefig(f"{save_path_names[0]}/class{i}_clean")
#                 f.write(f"Clean sum of class {i} evidence: {sum}\n")

#             if counter % 2 == 1: # odd counters are patched
#                 plt.savefig(f"{save_path_names[0]}/class{i}_adv")
#                 f.write(f"Adv sum of class {i} evidence: {sum}\n")
        
#             plt.close(fig)
#         if counter % 2 == 0:
#             f.write(f"\nClean max label: {max_class} and max sum: {max_sum}\n")
#         if counter % 2 == 1:
#             f.write(f"\nAdv max label: {max_class} and max sum: {max_sum}\n")
        
#         # Logit magnitude histogram
#         fig, ax = plt.subplots(1, 1)
#         ax.hist(logit_mgtds, bins = 40)
#         ax.set_xlabel("Logit Magnitude")
#         ax.set_ylabel("Count")
#         ax.set_title("Distribution of Local Logit Magnitudes")

#         if counter % 2 == 0: # even counters are clean
#             plt.savefig(f"{save_path_names[1]}/clean_hist")
        

#         if counter % 2 == 1: # odd counters are patched
#             plt.savefig(f"{save_path_names[1]}/adv_hist")
        
#         plt.close(fig)
#         # Boxplot
#         fig, ax = plt.subplots(1, 1)
#         ax.boxplot(logit_mgtds)
#         ax.set_ylabel("Logit Magnitudes")
#         ax.set_title(f"Boxplot of Local Logit Magnitudes {file_name}")

#         if counter % 2 == 0: # even counters are clean
            
#             plt.savefig(f"{save_path_names[2]}/clean_box")

#         if counter % 2 == 1: # odd counters are patched
#             plt.savefig(f"{save_path_names[2]}/adv_box")
#         plt.close(fig)

#         print(counter)
#         counter+=1

# print('Results for Derandomized Smoothing')
# print('Using band size ' + str(args.band_size) + ' with threshhold ' + str(0.2))
# print('Certifying For Patch ' +str(args.patch_size) + '*'+str(args.patch_size))
# print('Total images: ' + str(total))
# print('Correct: ' + str(correct) + ' (' + str((100.*correct)/total)+'%)')
# print('Certified Correct class: ' + str(cert_correct) + ' (' + str((100.*cert_correct)/total)+'%)')
# print('Certified Wrong class: ' + str(cert_incorrect) + ' (' + str((100.*cert_incorrect)/total)+'%)')


##############################################################################################################
# Part of code adapted from https://github.com/alevine0/patchSmoothing/blob/master/certify_imagenet_band.py
##############################################################################################################

from cProfile import label
from tkinter import image_names
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.special import rel_entr

import nets.dsresnet_imgnt as resnet_imgnt
import nets.dsresnet_cifar as resnet_cifar
from torchvision import datasets,transforms
from tqdm import tqdm
from utils.defense_utils import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from scipy.spatial import distance

import os
import argparse
import math
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--band_size', default=-1, type=int, help='size of each smoothing band')
parser.add_argument('--patch_size', default=-1, type=int, help='patch_size')
parser.add_argument('--thres', default=0.0, type=float, help='detection threshold for robus masking')
parser.add_argument('--dataset', default='imagenette', choices=('imagenette','imagenet','cifar', 'imagenette_pair_rn50', 'cifar_resnet18_ps5','cifar_resnet18_ps10', 'cifar_resnet18_ps2', 'cifar_resnet18_ps7', 'cifar_resnet18_ps4'),type=str,help="dataset")
parser.add_argument('--data_dir', default='data', type=str,help="path to data")

parser.add_argument('--skip', default=1,type=int, help='Number of images to skip')
parser.add_argument("--m",action='store_true',help="use robust masking")
parser.add_argument("--ds",action='store_true',help="use derandomized smoothing")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)
DATASET = args.dataset

device = 'cuda' #if torch.cuda.is_available() else 'cpu'

cudnn.benchmark = True

def get_dataset(ds,data_dir):
    if ds in ['imagenette','imagenet', 'imagenette_pair_rn50']:
        ds_dir=os.path.join(data_dir,'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset_ = datasets.ImageFolder(ds_dir, transforms.Compose([
                transforms.Resize((299,299)), #note that here input size if 299x299 instead of 224x224
                transforms.ToTensor(),
                normalize,
            ]))
    elif ds.startswith('cifar_resnet18'):
        ds_dir=os.path.join(data_dir,'val')
        transform_test = transforms.Compose([
                         transforms.ToTensor(),
                         #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                         ])
        dataset_ = datasets.ImageFolder(ds_dir, transforms.Compose([
                transforms.ToTensor(),
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
    net_name = "rn50"
    net = resnet_imgnt.resnet50()
    net = torch.nn.DataParallel(net)
    num_ftrs = net.module.fc.in_features
    net.module.fc = nn.Linear(num_ftrs, num_cls)  
    checkpoint = torch.load(os.path.join(MODEL_DIR,'ds_nette.pth'))
    args.band_size = args.band_size if args.band_size>0 else 25
    args.patch_size = args.patch_size if args.patch_size>0 else 42
elif DATASET == 'imagenet':
    net = resnet_imgnt.resnet50()
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load(os.path.join(MODEL_DIR,'ds_net.pth'))
    args.band_size = args.band_size if args.band_size>0 else 25
    args.patch_size = args.patch_size if args.patch_size>0 else 42
elif DATASET == 'cifar' or DATASET.startswith('cifar_resnet18'):
    net_name = "rn18"
    net = resnet_cifar.ResNet18()
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load(os.path.join(MODEL_DIR,'ds_cifar.pth'))
    args.band_size = args.band_size if args.band_size>0 else 4
    args.patch_size = args.patch_size if args.patch_size>0 else 5

print(args.band_size,args.patch_size)


net.load_state_dict(checkpoint['net'])

net = net.to(device)
net.eval()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def bs_least_outliers(logit_dict):
    logit_values_2d = np.array(list(logit_dict.values()))
    logit_values = logit_values_2d.flatten()

    med = np.median(logit_values)
    mad = np.abs(stats.median_absolute_deviation(logit_values))
    threshold = 2

    outlier = []
    for v in logit_values:
        t = (v-med)/mad
        if t > threshold:
            outlier.append(v)
        else:
            continue
    return outlier

def similarity(logit_dict):
    similarities = []
    for dict in list(logit_dict.values()):
        similarities.append(distance.jensenshannon(list(dict.values())[0], list(dict.values())[1]))
    return similarities
        



if args.ds:#ds
    correct = 0
    cert_correct = 0
    cert_incorrect = 0
    total = 0

    
    band_sizes = [2,5,10,15,20]
    logits_dict = {}
    clean_logits_2d = None
    clean_file_name = None

    counter = 0
    clean_prediction = None
    adversial_prediction = None
    

    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            if counter % 2 == 0 and counter!=0:
                print(f"Similarities: {similarity(logits_dict)}")
                filtered_similarities = [v for v in similarity(logits_dict) if not (math.isinf(v))]
                if len(filtered_similarities)>0:
                    print(f"Max similarities: {max(filtered_similarities)}")
                    print(f"Mean similarities: {np.mean(np.array(filtered_similarities))}")

            for band_size in band_sizes:
                sample_fname = val_loader.sampler.data_source.dataset.imgs[counter][0]
                sample_fname_list = sample_fname.split('/')
                file_name = sample_fname_list[-1]
                folder_name = sample_fname_list[-2]
                label_name = sample_fname_list[-3]
                print(f"file name: {file_name}")
                print(f"folder name: {folder_name}")
                print(f"label name: {label_name}")






                inputs, targets = inputs.to(device), targets.to(device)
                total += targets.size(0)
                predictions,  certyn, logits_2d, softmx_local_logits  = ds(inputs, net,band_size, args.patch_size, num_cls,threshold = 0.2)
                correct += (predictions.eq(targets)).sum().item()
                cert_correct += (predictions.eq(targets) & certyn).sum().item()
                cert_incorrect += (~predictions.eq(targets) & certyn).sum().item()

   

                logit_mgtds = np.linalg.norm(logits_2d, axis=1)
                logit_sums = np.sum(logits_2d, axis=1)

                # for i in range(logits_2d.shape[1]):
                #     sum = np.sum(logits_2d[:, i])
                #     sum = np.floor(sum)
                #     print(f"sum of class {i} evidence: {sum}")

                
                

                save_path = f"./plots/{net_name}_patch_plots/ps_{args.patch_size}/{label_name}/{folder_name}"

                save_path_names = [save_path+f"/class_evidence/bs_{band_size}", save_path+"/logit_mgtds_hist", save_path+"/logit_mgtds_box", save_path+f"/class_evidence/softmx/bs_{band_size}"]

                for path in save_path_names:
                    if not os.path.exists(path):
                        os.makedirs(path)

               


                if counter % 2 == 0:
                    clean_prediction  = predictions.cpu().numpy()[0]
                if counter % 2 == 1:
                    adversial_prediction  = predictions.cpu().numpy()[0]

                # Class evidence histograms
                for i in range(logits_2d.shape[1]):
                    
                    
                    if counter % 2== 1:

                        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)

                        # We can set the number of bins with the *bins* keyword argument.
                        axs[0].hist(clean_logits_2d[:, i], bins=20)
                        axs[1].hist(logits_2d[:, i], bins=20, color=["red"])

                        ax.set_xlabel(f"Class {i} Evidence")
                        ax.set_ylabel("Count")

                        fig.suptitle(f"Class {i} Logit Distributions with Label: {clean_prediction} Prediction: {adversial_prediction}")
                        axs[0].set_title(f"Clean")
                        axs[1].set_title(f"Adversarial")
                        
                        plt.savefig(f"{save_path_names[0]}/test_class_{i}_{file_name}")
                        plt.close(fig)


                    if counter % 2 == 0: # even counters are clean
                        
                        clean_logits_2d = np.copy(logits_2d)
                        clean_file_name = file_name
        
                    # we want to compare similarities between benign logit and malicious logit 
                    if clean_prediction == i and band_size==band_sizes[0]: 
    
                        if logits_dict.get(folder_name) is None:
                            logits_dict[folder_name] = dict({f"{file_name}_{clean_prediction}":logits_2d[:, i]})
                        else:
                            copy_dict = dict(logits_dict.get(folder_name))
                            copy_dict.update({f"{file_name}_{clean_prediction}":logits_2d[:, i]})
                            logits_dict[folder_name] = copy_dict
                    
                
                
                # if counter%2 == 1:
                #     for i in range(clean_logits_2d.shape[1]):
                #         sum = np.sum(logits_2d[:, i])
                #         sum = np.floor(sum)

                #         # we want to compare similarities between benign logit and malicious logit 
                #         if adversial_prediction == i and band_size==band_sizes[0]: 
                #             if logits_dict.get(folder_name+"_reverse") is None:
                #                 logits_dict[folder_name+"_reverse"] = dict({f"{file_name}_{adversial_prediction}":logits_2d[:, i]})
                                
                #                 copy_dict = dict(logits_dict.get(folder_name+"_reverse"))
                #                 copy_dict.update({f"{clean_file_name}_{adversial_prediction}":clean_logits_2d[:, i]})
                #                 logits_dict[folder_name+"_reverse"] = copy_dict
                # # Logit magnitude histogram
                fig, ax = plt.subplots(1, 1)
                ax.hist(logit_mgtds, bins = 40)
                ax.set_xlabel("Logit Magnitude")
                ax.set_ylabel("Count")
                ax.set_title(f"Distribution of Local Logit Magnitudes {file_name}")

                if counter % 2 == 0: # even counters are clean
                    plt.savefig(f"{save_path_names[1]}/bs_{band_size}_clean_hist")
                

                if counter % 2 == 1: # odd counters are patched
                    plt.savefig(f"{save_path_names[1]}/bs_{band_size}_adv_hist")
                
                plt.close(fig)
                # Boxplot
                fig, ax = plt.subplots(1, 1)
                ax.boxplot(logit_mgtds)
                ax.set_ylabel("Logit Magnitudes")
                ax.set_title(f"Boxplot of Local Logit Magnitudes {file_name}")

                if counter % 2 == 0: # even counters are clean
                    
                    plt.savefig(f"{save_path_names[2]}/bs_{band_size}_clean_box")

                if counter % 2 == 1: # odd counters are patched
                    plt.savefig(f"{save_path_names[2]}/bs_{band_size}_adv_box")
                plt.close(fig)

            
            counter+=1

            


            
            



    print('Results for Derandomized Smoothing')
    print('Using band size ' + str(args.band_size) + ' with threshhold ' + str(0.2))
    print('Certifying For Patch ' +str(args.patch_size) + '*'+str(args.patch_size))
    print('Total images: ' + str(total))
    print('Correct: ' + str(correct) + ' (' + str((100.*correct)/total)+'%)')
    print('Certified Correct class: ' + str(cert_correct) + ' (' + str((100.*cert_correct)/total)+'%)')
    print('Certified Wrong class: ' + str(cert_incorrect) + ' (' + str((100.*cert_incorrect)/total)+'%)')
    





if args.m:#mask-ds
    result_list=[]
    clean_corr_list=[]
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs = inputs.to(device)
            targets = targets.numpy()
            result,clean_corr = masking_ds(inputs,targets,net,args.band_size, args.patch_size,thres=args.thres)
            result_list+=result
            clean_corr_list+=clean_corr

    cases,cnt=np.unique(result_list,return_counts=True)
    print('Results for Mask-DS')
    print("Provable robust accuracy:",cnt[-1]/len(result_list) if len(cnt)==3 else 0)
    print("Clean accuracy with defense:",np.mean(clean_corr_list))
    print("------------------------------")
    print("Provable analysis cases (0: incorrect prediction; 1: vulnerable; 2: provably robust):",cases)
    print("Provable analysis breakdown:",cnt/len(result_list))
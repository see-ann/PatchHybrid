import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pytorch_cifar.models.resnet as resnet
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import PIL

import numpy as np

import os
import argparse
import utils_band as utils
from prog_bar import progress_bar
from patch_attacker_parallelized import PatchAttacker
parser = argparse.ArgumentParser(description='PyTorch CIFAR Attack')


parser.add_argument('--band_size', default=4, type=int, help='size of each smoothing band')
parser.add_argument('--size_of_attack', default=4, type=int, help='size of the attack')
parser.add_argument('--steps', default=150, type=int, help='Attack steps')
parser.add_argument('--randomizations', default=1, type=int, help='Number of random restarts')
parser.add_argument('--step_size', default=0.05, type=float, help='Attack step size')
parser.add_argument('--checkpoint', help='checkpoint')
parser.add_argument('--threshhold', default=0.3, type=float, help='threshold for smoothing abstain')
parser.add_argument('--model', default='resnet18', type=str, help='model')
parser.add_argument('--test', action='store_true', help='Use test set (vs validation)')
parser.add_argument('--skip', default=10,type=int, help='Number of images to skip')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mean_vec = [0.4914, 0.4822, 0.4465]
std_vec = [0.2023, 0.1994, 0.2010]

ds_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean_vec,std_vec),
])
val_indices = torch.load('validation.t7')
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=ds_transforms)
if (args.test):
    val_indices = list(set(range(len(testset))) - set(val_indices.numpy().tolist()))
val_indices = val_indices[::args.skip]
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset,val_indices), batch_size=1, shuffle=True, num_workers=2)


#              0       1      2      3    4         5      6       7        8       9
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
if (args.model == 'resnet50'):
    net = resnet.ResNet50()
elif (args.model == 'resnet18'):
    net = resnet.ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
#assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.checkpoint)

print(resume_file)
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['net'])
net.eval()
attacker = PatchAttacker(net, [0.,0.,0.],[1.,1.,1.], {
    'epsilon':1.0,
    'random_start':True,
    'steps':args.steps,
    'step_size':args.step_size,
    'block_size':args.band_size,
    'threshhold': args.threshhold,
    'num_classes':10,
    'patch_l':args.size_of_attack,
    'patch_w':args.size_of_attack
})

def test():
    global best_acc
    correctclean = 0
    correctattacked =0
    cert_correct = 0
    total = 0
    # For filepaths of saved images
    class_count = np.zeros(10, dtype = int)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        label = classes[targets.cpu().numpy()[0]]
        print("Targets: ")
        print(label)

        attacked = attacker.perturb(inputs,targets,float('inf'),random_count=args.randomizations)
        # finally correct inverse tranform
        # needed to apply mean and std transforms separately
        # ds_inverse_transforms = transforms.Compose([
        # transforms.Normalize(mean = [ 0., 0., 0. ],
        #                         std = [1/x for x in std_vec]),
        # transforms.Normalize(mean = [-x for x in mean_vec],
        #                     std = [ 1., 1., 1. ]),
        # ])

        # attacked_copy = ds_inverse_transforms(attacked)
        # clean_copy = ds_inverse_transforms(inputs)

        attacked_copy = attacked
        clean_copy = inputs



        predictionsclean,  certyn = utils.predict_and_certify(inputs, net,args.band_size, args.size_of_attack, 10,threshold =  args.threshhold)
        predictionsattacked,  certynx = utils.predict_and_certify(attacked, net,args.band_size, args.size_of_attack, 10,threshold =  args.threshhold)

        correctclean += (predictionsclean.eq(targets)).sum().item()
        correctattacked += (predictionsattacked.eq(targets)).sum().item()
        cert_correct += (predictionsclean.eq(targets) & certyn).sum().item()

        
        succ_attack = not predictionsattacked.eq(targets).cpu().numpy()[0]
        
        succ_clean_pred  = predictionsclean.eq(targets).cpu().numpy()[0]

        formatted_fn = f"class_{label}_img_{class_count[targets.cpu().numpy()[0]]}"
        save_path = f"./data/cifar_{args.model}_ps{args.size_of_attack}/val/{label}/{formatted_fn}"

        


        if succ_clean_pred:

            clean_pred = classes[predictionsclean.cpu().numpy()[0]]
            attacked_pred = classes[predictionsattacked.cpu().numpy()[0]]

            
            if succ_attack:

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    save_image(clean_copy, os.path.join(save_path, f"SUCC_clean_class_{label}_img_{class_count[targets.cpu().numpy()[0]]}_pred_{clean_pred}.png"))

                    save_image(attacked_copy, os.path.join(save_path, f"SUCC_patch_class_{label}_img_{class_count[targets.cpu().numpy()[0]]}_pred_{attacked_pred}.png"))

                print(f"Saved successful attack image")
            # else:

            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     if batch_idx % 2 == 0:
            #         save_image(clean_copy, os.path.join(save_path, f"UNSUCC_clean_class_{label}_img_{class_count[targets.cpu().numpy()[0]]}_pred_{clean_pred}.png"))

            #         save_image(attacked_copy, os.path.join(save_path, f"UNSUCC_patch_class_{label}_img_{class_count[targets.cpu().numpy()[0]]}_pred_{attacked_pred}.png"))
            
            class_count[targets.cpu().numpy()[0]] += 1

            

            





        progress_bar(batch_idx, len(testloader), 'Clean Acc: %.3f%% (%d/%d) Cert: %.3f%% (%d/%d) Adv Acc: %.3f%% (%d/%d)'  %  ((100.*correctclean)/total, correctclean, total, (100.*cert_correct)/total, cert_correct, total, (100.*correctattacked)/total, correctattacked, total))
    print('Using band size ' + str(args.band_size) + ' with threshhold ' + str(args.threshhold))
    print('Size of Attack Patch ' +str(args.size_of_attack) + '*'+str(args.size_of_attack))
    print('Total images: ' + str(total))
    print('Clean Correct: ' + str(correctclean) + ' (' + str((100.*correctclean)/total)+'%)')
    print('Certified: ' + str(cert_correct) + ' (' + str((100.*cert_correct)/total)+'%)')
    print('Attacked Correct: ' + str(correctattacked) + ' (' + str((100.*correctattacked)/total)+'%)')



test()

from __future__ import print_function
import torch
import numpy as np
import argparse
import os
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from torchvision import datasets, transforms
import time

from utils import Normalize

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('--model-type', default='resnet152', type=str)
parser.add_argument('--test-batch-size', type=int, default=24, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--model-dir', default='./saved_model',
                    help='directory of model for saving checkpoint')

args = parser.parse_args()

for arg in vars(args):
    print(arg, ':', getattr(args, arg))

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda:"+str(args.gpu) if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    loss = torch.nn.MSELoss(reduction='none')
    acc_sum, n = 0.0, 0
    test_l_sum, batch_count = 0.0, 0
    predict_status_arr = []
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            acc_sum += (output.argmax(dim=1) == y).float().sum().cpu().item()
            predict_status_arr.append((output.argmax(dim=1) == y).int().cpu().detach().numpy())
            # 计算泛化误差
            y_onehot = F.one_hot(y, 1000).float()
            l = loss(F.softmax(output, dim=1), y_onehot)
            test_l_sum += l.mean().cpu().item()

            n += y.shape[0]
            batch_count += 1
    predict_status_arr = np.concatenate(predict_status_arr, axis=0)
    return acc_sum / n, test_l_sum / batch_count, predict_status_arr

def calculate_and_save_mean_std_of_each_class(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device

    # 分类类别数
    num_classes = 1000
    # 正确分类的各类别logit值列表
    correct_logit_list = [[] for i in range(num_classes)]
    # 错误类别的各类别logit值列表
    wrong_logit_list = [[] for i in range(num_classes)]

    # 正确类别的各类别logit的均值和方差
    correct_logit_mean_std = [[] for i in range(num_classes)]
    # 错误类别的各类别logit的均值和方差
    wrong_logit_mean_std = [[] for i in range(num_classes)]

    # 执行分类
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            pred_arr = output.argmax(dim=1).cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            for i in range(len(pred_arr)):
                if y[i] == pred_arr[i] or True:
                    for j in range(num_classes):
                        if j == pred_arr[i]:
                            correct_logit_list[j].append(output[i][j])
                        else:
                            wrong_logit_list[j].append(output[i][j])

    # 计算均值和方差
    for i in range(num_classes):
        correct_logit_mean_std[i].extend([np.mean(correct_logit_list[i]), np.std(correct_logit_list[i])])
        wrong_logit_mean_std[i].extend([np.mean(wrong_logit_list[i]), np.std(wrong_logit_list[i])])

    return np.array(correct_logit_mean_std), np.array(wrong_logit_mean_std)

if __name__ == '__main__':
    # 加载替代模型
    if args.model_type == 'vgg16':
        model = models.vgg16()
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg16_checkpoint.pth')))
        model = nn.Sequential(
            Normalize(),
            model
        )
        model.to(device)
    elif args.model_type == 'resnet50':
        model = models.resnet50()
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'resnet50_checkpoint.pth')))
        model = nn.Sequential(
            Normalize(),
            model
        )
        model.to(device)
    elif args.model_type == 'resnet152':
        model = models.resnet152()
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'resnet152_checkpoint.pth')))
        model = nn.Sequential(
            Normalize(),
            model
        )
        model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageNet('./data/ImageNet2012', split='val', transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)

    acc, _, _ = evaluate_accuracy(data_loader, model, device=device)
    print(acc)
    start_time = time.time()

    correct_logit_mean_std, wrong_logit_mean_std = calculate_and_save_mean_std_of_each_class(data_iter=data_loader, net=model, device=device)

    end_time = time.time()

    print(correct_logit_mean_std)
    print(wrong_logit_mean_std)

    print("Spended time: ", end_time-start_time)

    np.save(os.path.join(args.model_dir, args.model_type+"_correct_logit_mean_std_test.npy"), correct_logit_mean_std)
    np.save(os.path.join(args.model_dir, args.model_type+"_wrong_logit_mean_std_test.npy"), wrong_logit_mean_std)

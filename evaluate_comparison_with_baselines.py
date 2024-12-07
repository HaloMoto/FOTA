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
from torchvision import transforms as T

from attack.FOTA import FOTA
from attack.DI_FOTA import DI_FOTA
from attack.Ada_DI_FOTA import Ada_DI_FOTA

from defense.bit_depth_reduction import BitDepthReduction
from defense.feature_distillation import FD_jpeg_encode
from defense.jpeg_compression import Jpeg_compresssion
from defense.nrp import NRP
from defense.randomization import Randomization
from defense.randomized_smoothing import Smooth

from utils import Normalize, Interpolate
from loader import ImageNet
from Selected_Imagenet_adv_test import SelectedImagenet_adv_test
from Selected_Imagenet_adv_test_RS import SelectedImagenet_adv_test_RS

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('--model-type', default='vgg16', type=str)
parser.add_argument('--test-batch-size', type=int, default=24, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--input_csv', default='./dataset/images.csv', type=str)
parser.add_argument('--input_dir', type=str, default='./dataset/images')
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

'''
MIFGSM 非目标攻击
'''
def Untarget_MIFGSM_test(model, device, test_loader, correct_logit_mean_std, wrong_logit_mean_std):
    # 生成对抗样本
    start_time = time.time()
    mifgsm = FOTA(model, eps=8.0 / 255, alpha=0.8 / 255, steps=10, decay=1.0, tau=1.0, kappa1=1.0, kappa2=1.0, num_classes=1000, correct_logit_mean_std=correct_logit_mean_std, wrong_logit_mean_std=wrong_logit_mean_std)
    data_adv_generated = []
    target_adv = []

    ASR_list = []
    transferability = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_adv = mifgsm(data, target, 1.0, 1.0)
        data_adv_generated.append(data_adv.detach().cpu().numpy())
        target_adv.extend(target.detach().cpu().numpy().tolist())
    end_time = time.time()
    time_spended = end_time - start_time
    # 加载对抗样本并进行预测
    data_adv_generated = np.concatenate(data_adv_generated, axis=0).transpose((0, 2, 3, 1))
    testset_adv_generated = SelectedImagenet_adv_test(data_adv_generated, target_adv)
    test_loader_adv_generated = torch.utils.data.DataLoader(testset_adv_generated, batch_size=args.test_batch_size,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=0)
    # VGG16
    success_attack_vgg16 = 0
    model_vgg16 = models.vgg16()
    model_vgg16.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg16_checkpoint.pth')))
    model_vgg16 = nn.Sequential(
        Normalize(),
        model_vgg16
    )
    model_vgg16.to(device)
    model_vgg16.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg16(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg16 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on VGG16: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg16,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg16 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg16':
        transferability += 100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset)
    # VGG19
    success_attack_vgg19 = 0
    model_vgg19 = models.vgg19()
    model_vgg19.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'vgg19_checkpoint.pth')))
    model_vgg19 = nn.Sequential(
        Normalize(),
        model_vgg19
    )
    model_vgg19.to(device)
    model_vgg19.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg19(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg19 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on VGG19: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg19,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg19 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg19':
        transferability += 100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset)
    # ResNet50
    success_attack_resnet50 = 0
    model_resnet50 = models.resnet50()
    model_resnet50.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet50_checkpoint.pth')))
    model_resnet50 = nn.Sequential(
        Normalize(),
        model_resnet50
    )
    model_resnet50.to(device)
    model_resnet50.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet50(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet50 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on ResNet50: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet50,
                                                              len(test_loader_adv_generated.dataset),
                                                              100. * success_attack_resnet50 / len(
                                                                  test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet50':
        transferability += 100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset)
    # ResNet152
    success_attack_resnet152 = 0
    model_resnet152 = models.resnet152()
    model_resnet152.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet152_checkpoint.pth')))
    model_resnet152 = nn.Sequential(
        Normalize(),
        model_resnet152
    )
    model_resnet152.to(device)
    model_resnet152.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet152(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet152 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on ResNet152: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet152,
                                                               len(test_loader_adv_generated.dataset),
                                                               100. * success_attack_resnet152 / len(
                                                                   test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet152':
        transferability += 100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset)
    # Inceptionv3
    success_attack_inceptionv3 = 0
    model_inceptionv3 = models.inception_v3()
    model_inceptionv3.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'inception_v3_checkpoint.pth')))
    model_inceptionv3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_inceptionv3
    )
    model_inceptionv3.to(device)
    model_inceptionv3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_inceptionv3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_inceptionv3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on Inceptionv3: ASR: {}/{} ({:.2f}%)'.format(success_attack_inceptionv3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_inceptionv3 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'inceptionv3':
        transferability += 100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset)
    # MobileNetv2
    success_attack_mobilenetv2 = 0
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2 = nn.Sequential(
        Normalize(),
        model_mobilenetv2
    )
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on MobileNetv2: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_mobilenetv2 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'mobilenetv2':
        transferability += 100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset)

    print('Transferability:{:.2f}%'.format(transferability/5))
    print('time spended:{}'.format(time_spended))

    # The transferability to the defenses
    ASR_against_defense = []
    # 按照R&P, Bit-Red, JPEG, FD, NRP, RS, advInc-v3, ensAdvIncRes-v2的顺序进行评估
    # R&P
    rp = Randomization(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_rp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = rp.random_resize_pad(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_rp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on MobileNetv2 with RP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rp,
                                                                           len(test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_rp / len(
                                                                               test_loader_adv_generated.dataset)))
    # Bit-Red
    bit_red = BitDepthReduction(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_bit_red = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = bit_red.bit_depth_reduction(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_bit_red += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on MobileNetv2 with Bit-Red: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_bit_red,
                                                                                len(test_loader_adv_generated.dataset),
                                                                                100. * success_attack_mobilenetv2_bit_red / len(
                                                                                    test_loader_adv_generated.dataset)))
    # JPEG
    jpeg = Jpeg_compresssion(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_jpeg = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = jpeg.jpegcompression(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_jpeg += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on MobileNetv2 with JPEG: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_jpeg,
                                                                             len(
                                                                                 test_loader_adv_generated.dataset),
                                                                             100. * success_attack_mobilenetv2_jpeg / len(
                                                                                 test_loader_adv_generated.dataset)))
    # FD
    success_attack_mobilenetv2_fd = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = data_adv.numpy()
        data_adv = FD_jpeg_encode(data_adv)
        data_adv = torch.from_numpy(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_fd += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on MobileNetv2 with FD: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_fd,
                                                                           len(
                                                                               test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_fd / len(
                                                                               test_loader_adv_generated.dataset)))
    # NRP
    netG = NRP(3, 3, 64, 23)
    netG.load_state_dict(torch.load("./defense/saved_model/NRP.pth"))
    netG = netG.to(device)
    netG.eval()
    success_attack_mobilenetv2_nrp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        eps_ = 16 / 255
        data_adv_m = data_adv + torch.randn_like(data_adv) * 0.05
        data_adv_m = torch.min(torch.max(data_adv_m, data_adv - eps_), data_adv + eps_)
        data_adv_m = torch.clamp(data_adv_m, 0.0, 1.0)
        with torch.no_grad():
            purified = netG(data_adv_m).detach()
            output = model_mobilenetv2(purified)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_nrp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on MobileNetv2 with NRP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_nrp,
                                                                            len(
                                                                                test_loader_adv_generated.dataset),
                                                                            100. * success_attack_mobilenetv2_nrp / len(
                                                                                test_loader_adv_generated.dataset)))
    # RS
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    rs = Smooth(model_mobilenetv2, 1000, 0.25)
    success_attack_mobilenetv2_rs = 0
    testset_adv_generated_RS = SelectedImagenet_adv_test_RS(data_adv_generated, target_adv)
    test_loader_adv_generated_RS = torch.utils.data.DataLoader(testset_adv_generated_RS, batch_size=1,
                                                               shuffle=False, pin_memory=True,
                                                               num_workers=0)
    for data_adv, target in test_loader_adv_generated_RS:
        data_adv = data_adv.to(device)
        output = rs.predict(data_adv, 100, 0.001, 100)
        success_attack_mobilenetv2_rs += int(output != target)
    print('MIFGSM Test on MobileNetv2 with RS: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rs,
                                                                           len(
                                                                               test_loader_adv_generated_RS.dataset),
                                                                           100. * success_attack_mobilenetv2_rs / len(
                                                                               test_loader_adv_generated_RS.dataset)))
    # advInc-v3
    success_attack_advInc_v3 = 0
    model_advInc_v3 = timm.create_model('adv_inception_v3', pretrained=True)
    model_advInc_v3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_advInc_v3
    )
    model_advInc_v3.to(device)
    model_advInc_v3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_advInc_v3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_advInc_v3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on advInc-v3: ASR: {}/{} ({:.2f}%)'.format(success_attack_advInc_v3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_advInc_v3 / len(
                                                                     test_loader_adv_generated.dataset)))
    # ensAdvIncRes-v2
    success_attack_ensAdvIncRes_v2 = 0
    model_ensAdvIncRes_v2 = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
    model_ensAdvIncRes_v2 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_ensAdvIncRes_v2
    )
    model_ensAdvIncRes_v2.to(device)
    model_ensAdvIncRes_v2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_ensAdvIncRes_v2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_ensAdvIncRes_v2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('MIFGSM Test on ensAdvIncRes-v2: ASR: {}/{} ({:.2f}%)'.format(success_attack_ensAdvIncRes_v2,
                                                                       len(test_loader_adv_generated.dataset),
                                                                       100. * success_attack_ensAdvIncRes_v2 / len(
                                                                           test_loader_adv_generated.dataset)))

    data_length = len(test_loader_adv_generated.dataset)
    data_length_rs = len(test_loader_adv_generated_RS.dataset)
    ASR_against_defense.append(
        [100. * success_attack_mobilenetv2_rp / data_length, 100. * success_attack_mobilenetv2_bit_red / data_length,
         100. * success_attack_mobilenetv2_jpeg / data_length,
         100. * success_attack_mobilenetv2_fd / data_length, 100. * success_attack_mobilenetv2_nrp / data_length,
         100. * success_attack_mobilenetv2_rs / data_length_rs,
         100. * success_attack_advInc_v3 / data_length, 100. * success_attack_ensAdvIncRes_v2 / data_length])
    print('Transferability to the defenses:{:.2f}%'.format(np.mean(ASR_against_defense)))

    return ASR_list, ASR_against_defense

'''
FOTA 非目标攻击
'''
def Untarget_FOTA_test(model, device, test_loader, correct_logit_mean_std, wrong_logit_mean_std):
    # 生成对抗样本
    start_time = time.time()
    fota = FOTA(model, eps=8.0 / 255, alpha=0.8 / 255, steps=10, decay=1.0, tau=1.5, kappa1=1.0, kappa2=0.9, num_classes=1000, correct_logit_mean_std=correct_logit_mean_std, wrong_logit_mean_std=wrong_logit_mean_std)
    data_adv_generated = []
    target_adv = []

    ASR_list = []
    transferability = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_adv = fota(data, target, 1.0, 0.9)
        data_adv_generated.append(data_adv.detach().cpu().numpy())
        target_adv.extend(target.detach().cpu().numpy().tolist())
    end_time = time.time()
    time_spended = end_time - start_time
    # 加载对抗样本并进行预测
    data_adv_generated = np.concatenate(data_adv_generated, axis=0).transpose((0, 2, 3, 1))
    testset_adv_generated = SelectedImagenet_adv_test(data_adv_generated, target_adv)
    test_loader_adv_generated = torch.utils.data.DataLoader(testset_adv_generated, batch_size=args.test_batch_size,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=0)
    # VGG16
    success_attack_vgg16 = 0
    model_vgg16 = models.vgg16()
    model_vgg16.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg16_checkpoint.pth')))
    model_vgg16 = nn.Sequential(
        Normalize(),
        model_vgg16
    )
    model_vgg16.to(device)
    model_vgg16.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg16(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg16 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on VGG16: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg16,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg16 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg16':
        transferability += 100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset)
    # VGG19
    success_attack_vgg19 = 0
    model_vgg19 = models.vgg19()
    model_vgg19.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'vgg19_checkpoint.pth')))
    model_vgg19 = nn.Sequential(
        Normalize(),
        model_vgg19
    )
    model_vgg19.to(device)
    model_vgg19.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg19(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg19 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on VGG19: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg19,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg19 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg19':
        transferability += 100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset)
    # ResNet50
    success_attack_resnet50 = 0
    model_resnet50 = models.resnet50()
    model_resnet50.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet50_checkpoint.pth')))
    model_resnet50 = nn.Sequential(
        Normalize(),
        model_resnet50
    )
    model_resnet50.to(device)
    model_resnet50.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet50(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet50 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on ResNet50: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet50,
                                                              len(test_loader_adv_generated.dataset),
                                                              100. * success_attack_resnet50 / len(
                                                                  test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet50':
        transferability += 100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset)
    # ResNet152
    success_attack_resnet152 = 0
    model_resnet152 = models.resnet152()
    model_resnet152.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet152_checkpoint.pth')))
    model_resnet152 = nn.Sequential(
        Normalize(),
        model_resnet152
    )
    model_resnet152.to(device)
    model_resnet152.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet152(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet152 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on ResNet152: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet152,
                                                               len(test_loader_adv_generated.dataset),
                                                               100. * success_attack_resnet152 / len(
                                                                   test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet152':
        transferability += 100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset)
    # Inceptionv3
    success_attack_inceptionv3 = 0
    model_inceptionv3 = models.inception_v3()
    model_inceptionv3.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'inception_v3_checkpoint.pth')))
    model_inceptionv3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_inceptionv3
    )
    model_inceptionv3.to(device)
    model_inceptionv3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_inceptionv3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_inceptionv3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on Inceptionv3: ASR: {}/{} ({:.2f}%)'.format(success_attack_inceptionv3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_inceptionv3 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'inceptionv3':
        transferability += 100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset)
    # MobileNetv2
    success_attack_mobilenetv2 = 0
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2 = nn.Sequential(
        Normalize(),
        model_mobilenetv2
    )
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on MobileNetv2: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_mobilenetv2 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'mobilenetv2':
        transferability += 100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset)

    print('Transferability:{:.2f}%'.format(transferability/5))
    print('time spended:{}'.format(time_spended))

    # The transferability to the defenses
    ASR_against_defense = []
    # 按照R&P, Bit-Red, JPEG, FD, NRP, RS, advInc-v3, ensAdvIncRes-v2的顺序进行评估
    # R&P
    rp = Randomization(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_rp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = rp.random_resize_pad(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_rp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on MobileNetv2 with RP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rp,
                                                                           len(test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_rp / len(
                                                                               test_loader_adv_generated.dataset)))
    # Bit-Red
    bit_red = BitDepthReduction(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_bit_red = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = bit_red.bit_depth_reduction(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_bit_red += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on MobileNetv2 with Bit-Red: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_bit_red,
                                                                                len(test_loader_adv_generated.dataset),
                                                                                100. * success_attack_mobilenetv2_bit_red / len(
                                                                                    test_loader_adv_generated.dataset)))
    # JPEG
    jpeg = Jpeg_compresssion(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_jpeg = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = jpeg.jpegcompression(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_jpeg += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on MobileNetv2 with JPEG: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_jpeg,
                                                                             len(
                                                                                 test_loader_adv_generated.dataset),
                                                                             100. * success_attack_mobilenetv2_jpeg / len(
                                                                                 test_loader_adv_generated.dataset)))
    # FD
    success_attack_mobilenetv2_fd = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = data_adv.numpy()
        data_adv = FD_jpeg_encode(data_adv)
        data_adv = torch.from_numpy(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_fd += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on MobileNetv2 with FD: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_fd,
                                                                           len(
                                                                               test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_fd / len(
                                                                               test_loader_adv_generated.dataset)))
    # NRP
    netG = NRP(3, 3, 64, 23)
    netG.load_state_dict(torch.load("./defense/saved_model/NRP.pth"))
    netG = netG.to(device)
    netG.eval()
    success_attack_mobilenetv2_nrp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        eps_ = 16 / 255
        data_adv_m = data_adv + torch.randn_like(data_adv) * 0.05
        data_adv_m = torch.min(torch.max(data_adv_m, data_adv - eps_), data_adv + eps_)
        data_adv_m = torch.clamp(data_adv_m, 0.0, 1.0)
        with torch.no_grad():
            purified = netG(data_adv_m).detach()
            output = model_mobilenetv2(purified)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_nrp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on MobileNetv2 with NRP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_nrp,
                                                                            len(
                                                                                test_loader_adv_generated.dataset),
                                                                            100. * success_attack_mobilenetv2_nrp / len(
                                                                                test_loader_adv_generated.dataset)))
    # RS
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    rs = Smooth(model_mobilenetv2, 1000, 0.25)
    success_attack_mobilenetv2_rs = 0
    testset_adv_generated_RS = SelectedImagenet_adv_test_RS(data_adv_generated, target_adv)
    test_loader_adv_generated_RS = torch.utils.data.DataLoader(testset_adv_generated_RS, batch_size=1,
                                                               shuffle=False, pin_memory=True,
                                                               num_workers=0)
    for data_adv, target in test_loader_adv_generated_RS:
        data_adv = data_adv.to(device)
        output = rs.predict(data_adv, 100, 0.001, 100)
        success_attack_mobilenetv2_rs += int(output != target)
    print('FOTA Test on MobileNetv2 with RS: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rs,
                                                                           len(
                                                                               test_loader_adv_generated_RS.dataset),
                                                                           100. * success_attack_mobilenetv2_rs / len(
                                                                               test_loader_adv_generated_RS.dataset)))
    # advInc-v3
    success_attack_advInc_v3 = 0
    model_advInc_v3 = timm.create_model('adv_inception_v3', pretrained=True)
    model_advInc_v3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_advInc_v3
    )
    model_advInc_v3.to(device)
    model_advInc_v3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_advInc_v3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_advInc_v3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on advInc-v3: ASR: {}/{} ({:.2f}%)'.format(success_attack_advInc_v3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_advInc_v3 / len(
                                                                     test_loader_adv_generated.dataset)))
    # ensAdvIncRes-v2
    success_attack_ensAdvIncRes_v2 = 0
    model_ensAdvIncRes_v2 = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
    model_ensAdvIncRes_v2 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_ensAdvIncRes_v2
    )
    model_ensAdvIncRes_v2.to(device)
    model_ensAdvIncRes_v2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_ensAdvIncRes_v2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_ensAdvIncRes_v2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('FOTA Test on ensAdvIncRes-v2: ASR: {}/{} ({:.2f}%)'.format(success_attack_ensAdvIncRes_v2,
                                                                       len(test_loader_adv_generated.dataset),
                                                                       100. * success_attack_ensAdvIncRes_v2 / len(
                                                                           test_loader_adv_generated.dataset)))

    data_length = len(test_loader_adv_generated.dataset)
    data_length_rs = len(test_loader_adv_generated_RS.dataset)
    ASR_against_defense.append(
        [100. * success_attack_mobilenetv2_rp / data_length, 100. * success_attack_mobilenetv2_bit_red / data_length,
         100. * success_attack_mobilenetv2_jpeg / data_length,
         100. * success_attack_mobilenetv2_fd / data_length, 100. * success_attack_mobilenetv2_nrp / data_length,
         100. * success_attack_mobilenetv2_rs / data_length_rs,
         100. * success_attack_advInc_v3 / data_length, 100. * success_attack_ensAdvIncRes_v2 / data_length])
    print('Transferability to the defenses:{:.2f}%'.format(np.mean(ASR_against_defense)))

    return ASR_list, ASR_against_defense

'''
DI_MIFGSM 非目标攻击
'''
def Untarget_DI_MIFGSM_test(model, device, test_loader, correct_logit_mean_std, wrong_logit_mean_std):
    # 生成对抗样本
    start_time = time.time()
    di_mifgsm = DI_FOTA(model, eps=8.0 / 255, alpha=0.8 / 255, steps=10, decay=1.0, tau=1.0, kappa1=1.0, kappa2=1.0, num_classes=1000, correct_logit_mean_std=correct_logit_mean_std, wrong_logit_mean_std=wrong_logit_mean_std)
    data_adv_generated = []
    target_adv = []

    ASR_list = []
    transferability = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_adv = di_mifgsm(data, target, 1.0, 1.0)
        data_adv_generated.append(data_adv.detach().cpu().numpy())
        target_adv.extend(target.detach().cpu().numpy().tolist())
    end_time = time.time()
    time_spended = end_time - start_time
    # 加载对抗样本并进行预测
    data_adv_generated = np.concatenate(data_adv_generated, axis=0).transpose((0, 2, 3, 1))
    testset_adv_generated = SelectedImagenet_adv_test(data_adv_generated, target_adv)
    test_loader_adv_generated = torch.utils.data.DataLoader(testset_adv_generated, batch_size=args.test_batch_size,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=0)
    # VGG16
    success_attack_vgg16 = 0
    model_vgg16 = models.vgg16()
    model_vgg16.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg16_checkpoint.pth')))
    model_vgg16 = nn.Sequential(
        Normalize(),
        model_vgg16
    )
    model_vgg16.to(device)
    model_vgg16.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg16(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg16 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on VGG16: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg16,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg16 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg16':
        transferability += 100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset)
    # VGG19
    success_attack_vgg19 = 0
    model_vgg19 = models.vgg19()
    model_vgg19.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'vgg19_checkpoint.pth')))
    model_vgg19 = nn.Sequential(
        Normalize(),
        model_vgg19
    )
    model_vgg19.to(device)
    model_vgg19.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg19(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg19 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on VGG19: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg19,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg19 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg19':
        transferability += 100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset)
    # ResNet50
    success_attack_resnet50 = 0
    model_resnet50 = models.resnet50()
    model_resnet50.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet50_checkpoint.pth')))
    model_resnet50 = nn.Sequential(
        Normalize(),
        model_resnet50
    )
    model_resnet50.to(device)
    model_resnet50.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet50(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet50 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on ResNet50: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet50,
                                                              len(test_loader_adv_generated.dataset),
                                                              100. * success_attack_resnet50 / len(
                                                                  test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet50':
        transferability += 100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset)
    # ResNet152
    success_attack_resnet152 = 0
    model_resnet152 = models.resnet152()
    model_resnet152.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet152_checkpoint.pth')))
    model_resnet152 = nn.Sequential(
        Normalize(),
        model_resnet152
    )
    model_resnet152.to(device)
    model_resnet152.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet152(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet152 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on ResNet152: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet152,
                                                               len(test_loader_adv_generated.dataset),
                                                               100. * success_attack_resnet152 / len(
                                                                   test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet152':
        transferability += 100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset)
    # Inceptionv3
    success_attack_inceptionv3 = 0
    model_inceptionv3 = models.inception_v3()
    model_inceptionv3.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'inception_v3_checkpoint.pth')))
    model_inceptionv3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_inceptionv3
    )
    model_inceptionv3.to(device)
    model_inceptionv3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_inceptionv3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_inceptionv3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on Inceptionv3: ASR: {}/{} ({:.2f}%)'.format(success_attack_inceptionv3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_inceptionv3 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'inceptionv3':
        transferability += 100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset)
    # MobileNetv2
    success_attack_mobilenetv2 = 0
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2 = nn.Sequential(
        Normalize(),
        model_mobilenetv2
    )
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on MobileNetv2: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_mobilenetv2 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'mobilenetv2':
        transferability += 100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset)

    print('Transferability:{:.2f}%'.format(transferability/5))
    print('time spended:{}'.format(time_spended))

    # The transferability to the defenses
    ASR_against_defense = []
    # 按照R&P, Bit-Red, JPEG, FD, NRP, RS, advInc-v3, ensAdvIncRes-v2的顺序进行评估
    # R&P
    rp = Randomization(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_rp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = rp.random_resize_pad(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_rp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on MobileNetv2 with RP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rp,
                                                                           len(test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_rp / len(
                                                                               test_loader_adv_generated.dataset)))
    # Bit-Red
    bit_red = BitDepthReduction(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_bit_red = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = bit_red.bit_depth_reduction(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_bit_red += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on MobileNetv2 with Bit-Red: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_bit_red,
                                                                                len(test_loader_adv_generated.dataset),
                                                                                100. * success_attack_mobilenetv2_bit_red / len(
                                                                                    test_loader_adv_generated.dataset)))
    # JPEG
    jpeg = Jpeg_compresssion(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_jpeg = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = jpeg.jpegcompression(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_jpeg += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on MobileNetv2 with JPEG: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_jpeg,
                                                                             len(
                                                                                 test_loader_adv_generated.dataset),
                                                                             100. * success_attack_mobilenetv2_jpeg / len(
                                                                                 test_loader_adv_generated.dataset)))
    # FD
    success_attack_mobilenetv2_fd = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = data_adv.numpy()
        data_adv = FD_jpeg_encode(data_adv)
        data_adv = torch.from_numpy(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_fd += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on MobileNetv2 with FD: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_fd,
                                                                           len(
                                                                               test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_fd / len(
                                                                               test_loader_adv_generated.dataset)))
    # NRP
    netG = NRP(3, 3, 64, 23)
    netG.load_state_dict(torch.load("./defense/saved_model/NRP.pth"))
    netG = netG.to(device)
    netG.eval()
    success_attack_mobilenetv2_nrp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        eps_ = 16 / 255
        data_adv_m = data_adv + torch.randn_like(data_adv) * 0.05
        data_adv_m = torch.min(torch.max(data_adv_m, data_adv - eps_), data_adv + eps_)
        data_adv_m = torch.clamp(data_adv_m, 0.0, 1.0)
        with torch.no_grad():
            purified = netG(data_adv_m).detach()
            output = model_mobilenetv2(purified)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_nrp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on MobileNetv2 with NRP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_nrp,
                                                                            len(
                                                                                test_loader_adv_generated.dataset),
                                                                            100. * success_attack_mobilenetv2_nrp / len(
                                                                                test_loader_adv_generated.dataset)))
    # RS
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    rs = Smooth(model_mobilenetv2, 1000, 0.25)
    success_attack_mobilenetv2_rs = 0
    testset_adv_generated_RS = SelectedImagenet_adv_test_RS(data_adv_generated, target_adv)
    test_loader_adv_generated_RS = torch.utils.data.DataLoader(testset_adv_generated_RS, batch_size=1,
                                                               shuffle=False, pin_memory=True,
                                                               num_workers=0)
    for data_adv, target in test_loader_adv_generated_RS:
        data_adv = data_adv.to(device)
        output = rs.predict(data_adv, 100, 0.001, 100)
        success_attack_mobilenetv2_rs += int(output != target)
    print('DI_MIFGSM Test on MobileNetv2 with RS: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rs,
                                                                           len(
                                                                               test_loader_adv_generated_RS.dataset),
                                                                           100. * success_attack_mobilenetv2_rs / len(
                                                                               test_loader_adv_generated_RS.dataset)))
    # advInc-v3
    success_attack_advInc_v3 = 0
    model_advInc_v3 = timm.create_model('adv_inception_v3', pretrained=True)
    model_advInc_v3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_advInc_v3
    )
    model_advInc_v3.to(device)
    model_advInc_v3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_advInc_v3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_advInc_v3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on advInc-v3: ASR: {}/{} ({:.2f}%)'.format(success_attack_advInc_v3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_advInc_v3 / len(
                                                                     test_loader_adv_generated.dataset)))
    # ensAdvIncRes-v2
    success_attack_ensAdvIncRes_v2 = 0
    model_ensAdvIncRes_v2 = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
    model_ensAdvIncRes_v2 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_ensAdvIncRes_v2
    )
    model_ensAdvIncRes_v2.to(device)
    model_ensAdvIncRes_v2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_ensAdvIncRes_v2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_ensAdvIncRes_v2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_MIFGSM Test on ensAdvIncRes-v2: ASR: {}/{} ({:.2f}%)'.format(success_attack_ensAdvIncRes_v2,
                                                                       len(test_loader_adv_generated.dataset),
                                                                       100. * success_attack_ensAdvIncRes_v2 / len(
                                                                           test_loader_adv_generated.dataset)))

    data_length = len(test_loader_adv_generated.dataset)
    data_length_rs = len(test_loader_adv_generated_RS.dataset)
    ASR_against_defense.append(
        [100. * success_attack_mobilenetv2_rp / data_length, 100. * success_attack_mobilenetv2_bit_red / data_length,
         100. * success_attack_mobilenetv2_jpeg / data_length,
         100. * success_attack_mobilenetv2_fd / data_length, 100. * success_attack_mobilenetv2_nrp / data_length,
         100. * success_attack_mobilenetv2_rs / data_length_rs,
         100. * success_attack_advInc_v3 / data_length, 100. * success_attack_ensAdvIncRes_v2 / data_length])
    print('Transferability to the defenses:{:.2f}%'.format(np.mean(ASR_against_defense)))

    return ASR_list, ASR_against_defense

'''
DI_FOTA 非目标攻击
'''
def Untarget_DI_FOTA_test(model, device, test_loader, correct_logit_mean_std, wrong_logit_mean_std):
    # 生成对抗样本
    start_time = time.time()
    di_fota = DI_FOTA(model, eps=8.0 / 255, alpha=0.8 / 255, steps=10, decay=1.0, tau=1.5, kappa1=1.0, kappa2=0.9, num_classes=1000, correct_logit_mean_std=correct_logit_mean_std, wrong_logit_mean_std=wrong_logit_mean_std)
    data_adv_generated = []
    target_adv = []

    ASR_list = []
    transferability = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_adv = di_fota(data, target, 1.0, 0.9)
        data_adv_generated.append(data_adv.detach().cpu().numpy())
        target_adv.extend(target.detach().cpu().numpy().tolist())
    end_time = time.time()
    time_spended = end_time - start_time
    # 加载对抗样本并进行预测
    data_adv_generated = np.concatenate(data_adv_generated, axis=0).transpose((0, 2, 3, 1))
    testset_adv_generated = SelectedImagenet_adv_test(data_adv_generated, target_adv)
    test_loader_adv_generated = torch.utils.data.DataLoader(testset_adv_generated, batch_size=args.test_batch_size,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=0)
    # VGG16
    success_attack_vgg16 = 0
    model_vgg16 = models.vgg16()
    model_vgg16.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg16_checkpoint.pth')))
    model_vgg16 = nn.Sequential(
        Normalize(),
        model_vgg16
    )
    model_vgg16.to(device)
    model_vgg16.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg16(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg16 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on VGG16: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg16,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg16 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg16':
        transferability += 100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset)
    # VGG19
    success_attack_vgg19 = 0
    model_vgg19 = models.vgg19()
    model_vgg19.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'vgg19_checkpoint.pth')))
    model_vgg19 = nn.Sequential(
        Normalize(),
        model_vgg19
    )
    model_vgg19.to(device)
    model_vgg19.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg19(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg19 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on VGG19: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg19,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg19 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg19':
        transferability += 100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset)
    # ResNet50
    success_attack_resnet50 = 0
    model_resnet50 = models.resnet50()
    model_resnet50.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet50_checkpoint.pth')))
    model_resnet50 = nn.Sequential(
        Normalize(),
        model_resnet50
    )
    model_resnet50.to(device)
    model_resnet50.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet50(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet50 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on ResNet50: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet50,
                                                              len(test_loader_adv_generated.dataset),
                                                              100. * success_attack_resnet50 / len(
                                                                  test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet50':
        transferability += 100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset)
    # ResNet152
    success_attack_resnet152 = 0
    model_resnet152 = models.resnet152()
    model_resnet152.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet152_checkpoint.pth')))
    model_resnet152 = nn.Sequential(
        Normalize(),
        model_resnet152
    )
    model_resnet152.to(device)
    model_resnet152.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet152(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet152 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on ResNet152: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet152,
                                                               len(test_loader_adv_generated.dataset),
                                                               100. * success_attack_resnet152 / len(
                                                                   test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet152':
        transferability += 100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset)
    # Inceptionv3
    success_attack_inceptionv3 = 0
    model_inceptionv3 = models.inception_v3()
    model_inceptionv3.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'inception_v3_checkpoint.pth')))
    model_inceptionv3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_inceptionv3
    )
    model_inceptionv3.to(device)
    model_inceptionv3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_inceptionv3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_inceptionv3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on Inceptionv3: ASR: {}/{} ({:.2f}%)'.format(success_attack_inceptionv3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_inceptionv3 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'inceptionv3':
        transferability += 100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset)
    # MobileNetv2
    success_attack_mobilenetv2 = 0
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2 = nn.Sequential(
        Normalize(),
        model_mobilenetv2
    )
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on MobileNetv2: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_mobilenetv2 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'mobilenetv2':
        transferability += 100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset)

    print('Transferability:{:.2f}%'.format(transferability/5))
    print('time spended:{}'.format(time_spended))

    # The transferability to the defenses
    ASR_against_defense = []
    # 按照R&P, Bit-Red, JPEG, FD, NRP, RS, advInc-v3, ensAdvIncRes-v2的顺序进行评估
    # R&P
    rp = Randomization(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_rp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = rp.random_resize_pad(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_rp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on MobileNetv2 with RP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rp,
                                                                           len(test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_rp / len(
                                                                               test_loader_adv_generated.dataset)))
    # Bit-Red
    bit_red = BitDepthReduction(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_bit_red = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = bit_red.bit_depth_reduction(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_bit_red += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on MobileNetv2 with Bit-Red: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_bit_red,
                                                                                len(test_loader_adv_generated.dataset),
                                                                                100. * success_attack_mobilenetv2_bit_red / len(
                                                                                    test_loader_adv_generated.dataset)))
    # JPEG
    jpeg = Jpeg_compresssion(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_jpeg = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = jpeg.jpegcompression(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_jpeg += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on MobileNetv2 with JPEG: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_jpeg,
                                                                             len(
                                                                                 test_loader_adv_generated.dataset),
                                                                             100. * success_attack_mobilenetv2_jpeg / len(
                                                                                 test_loader_adv_generated.dataset)))
    # FD
    success_attack_mobilenetv2_fd = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = data_adv.numpy()
        data_adv = FD_jpeg_encode(data_adv)
        data_adv = torch.from_numpy(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_fd += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on MobileNetv2 with FD: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_fd,
                                                                           len(
                                                                               test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_fd / len(
                                                                               test_loader_adv_generated.dataset)))
    # NRP
    netG = NRP(3, 3, 64, 23)
    netG.load_state_dict(torch.load("./defense/saved_model/NRP.pth"))
    netG = netG.to(device)
    netG.eval()
    success_attack_mobilenetv2_nrp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        eps_ = 16 / 255
        data_adv_m = data_adv + torch.randn_like(data_adv) * 0.05
        data_adv_m = torch.min(torch.max(data_adv_m, data_adv - eps_), data_adv + eps_)
        data_adv_m = torch.clamp(data_adv_m, 0.0, 1.0)
        with torch.no_grad():
            purified = netG(data_adv_m).detach()
            output = model_mobilenetv2(purified)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_nrp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on MobileNetv2 with NRP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_nrp,
                                                                            len(
                                                                                test_loader_adv_generated.dataset),
                                                                            100. * success_attack_mobilenetv2_nrp / len(
                                                                                test_loader_adv_generated.dataset)))
    # RS
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    rs = Smooth(model_mobilenetv2, 1000, 0.25)
    success_attack_mobilenetv2_rs = 0
    testset_adv_generated_RS = SelectedImagenet_adv_test_RS(data_adv_generated, target_adv)
    test_loader_adv_generated_RS = torch.utils.data.DataLoader(testset_adv_generated_RS, batch_size=1,
                                                               shuffle=False, pin_memory=True,
                                                               num_workers=0)
    for data_adv, target in test_loader_adv_generated_RS:
        data_adv = data_adv.to(device)
        output = rs.predict(data_adv, 100, 0.001, 100)
        success_attack_mobilenetv2_rs += int(output != target)
    print('DI_FOTA Test on MobileNetv2 with RS: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rs,
                                                                           len(
                                                                               test_loader_adv_generated_RS.dataset),
                                                                           100. * success_attack_mobilenetv2_rs / len(
                                                                               test_loader_adv_generated_RS.dataset)))
    # advInc-v3
    success_attack_advInc_v3 = 0
    model_advInc_v3 = timm.create_model('adv_inception_v3', pretrained=True)
    model_advInc_v3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_advInc_v3
    )
    model_advInc_v3.to(device)
    model_advInc_v3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_advInc_v3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_advInc_v3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on advInc-v3: ASR: {}/{} ({:.2f}%)'.format(success_attack_advInc_v3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_advInc_v3 / len(
                                                                     test_loader_adv_generated.dataset)))
    # ensAdvIncRes-v2
    success_attack_ensAdvIncRes_v2 = 0
    model_ensAdvIncRes_v2 = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
    model_ensAdvIncRes_v2 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_ensAdvIncRes_v2
    )
    model_ensAdvIncRes_v2.to(device)
    model_ensAdvIncRes_v2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_ensAdvIncRes_v2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_ensAdvIncRes_v2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('DI_FOTA Test on ensAdvIncRes-v2: ASR: {}/{} ({:.2f}%)'.format(success_attack_ensAdvIncRes_v2,
                                                                       len(test_loader_adv_generated.dataset),
                                                                       100. * success_attack_ensAdvIncRes_v2 / len(
                                                                           test_loader_adv_generated.dataset)))

    data_length = len(test_loader_adv_generated.dataset)
    data_length_rs = len(test_loader_adv_generated_RS.dataset)
    ASR_against_defense.append(
        [100. * success_attack_mobilenetv2_rp / data_length, 100. * success_attack_mobilenetv2_bit_red / data_length,
         100. * success_attack_mobilenetv2_jpeg / data_length,
         100. * success_attack_mobilenetv2_fd / data_length, 100. * success_attack_mobilenetv2_nrp / data_length,
         100. * success_attack_mobilenetv2_rs / data_length_rs,
         100. * success_attack_advInc_v3 / data_length, 100. * success_attack_ensAdvIncRes_v2 / data_length])
    print('Transferability to the defenses:{:.2f}%'.format(np.mean(ASR_against_defense)))

    return ASR_list, ASR_against_defense

'''
Ada_DI_FOTA 非目标攻击
'''
def Untarget_Ada_DI_FOTA_test(model, device, test_loader, correct_logit_mean_std, wrong_logit_mean_std):
    # 生成对抗样本
    start_time = time.time()
    ada_di_fota = Ada_DI_FOTA(model, eps=8.0 / 255, alpha=0.8 / 255, max_steps=100, membership_degree_threshold=1-1e-10, decay=1.0, tau=1.5, kappa1=1.0, kappa2=1.0, num_classes=1000, correct_logit_mean_std=correct_logit_mean_std, wrong_logit_mean_std=wrong_logit_mean_std)
    data_adv_generated = []
    target_adv = []
    iteration_statistical_arr = []
    membership_degree_arr = []

    ASR_list = []
    transferability = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_adv, iteration_statistical, membership_degrees = ada_di_fota(data, target, 1.0, 0.9)
        iteration_statistical_arr.extend(iteration_statistical)
        membership_degree_arr.extend(membership_degrees)
        data_adv_generated.append(data_adv.detach().cpu().numpy())
        target_adv.extend(target.detach().cpu().numpy().tolist())
    print("The average number of iteration: ", np.array(iteration_statistical_arr).mean())
    print("The average membership degree: ", np.array(membership_degree_arr).mean())
    end_time = time.time()
    time_spended = end_time - start_time
    # 加载对抗样本并进行预测
    data_adv_generated = np.concatenate(data_adv_generated, axis=0).transpose((0, 2, 3, 1))
    testset_adv_generated = SelectedImagenet_adv_test(data_adv_generated, target_adv)
    test_loader_adv_generated = torch.utils.data.DataLoader(testset_adv_generated, batch_size=args.test_batch_size,
                                                            shuffle=False, pin_memory=True,
                                                            num_workers=0)
    # VGG16
    success_attack_vgg16 = 0
    model_vgg16 = models.vgg16()
    model_vgg16.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg16_checkpoint.pth')))
    model_vgg16 = nn.Sequential(
        Normalize(),
        model_vgg16
    )
    model_vgg16.to(device)
    model_vgg16.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg16(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg16 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on VGG16: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg16,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg16 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg16':
        transferability += 100. * success_attack_vgg16 / len(test_loader_adv_generated.dataset)
    # VGG19
    success_attack_vgg19 = 0
    model_vgg19 = models.vgg19()
    model_vgg19.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'vgg19_checkpoint.pth')))
    model_vgg19 = nn.Sequential(
        Normalize(),
        model_vgg19
    )
    model_vgg19.to(device)
    model_vgg19.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_vgg19(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_vgg19 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on VGG19: ASR: {}/{} ({:.2f}%)'.format(success_attack_vgg19,
                                                           len(test_loader_adv_generated.dataset),
                                                           100. * success_attack_vgg19 / len(
                                                               test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'vgg19':
        transferability += 100. * success_attack_vgg19 / len(test_loader_adv_generated.dataset)
    # ResNet50
    success_attack_resnet50 = 0
    model_resnet50 = models.resnet50()
    model_resnet50.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet50_checkpoint.pth')))
    model_resnet50 = nn.Sequential(
        Normalize(),
        model_resnet50
    )
    model_resnet50.to(device)
    model_resnet50.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet50(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet50 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on ResNet50: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet50,
                                                              len(test_loader_adv_generated.dataset),
                                                              100. * success_attack_resnet50 / len(
                                                                  test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet50':
        transferability += 100. * success_attack_resnet50 / len(test_loader_adv_generated.dataset)
    # ResNet152
    success_attack_resnet152 = 0
    model_resnet152 = models.resnet152()
    model_resnet152.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'resnet152_checkpoint.pth')))
    model_resnet152 = nn.Sequential(
        Normalize(),
        model_resnet152
    )
    model_resnet152.to(device)
    model_resnet152.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_resnet152(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_resnet152 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on ResNet152: ASR: {}/{} ({:.2f}%)'.format(success_attack_resnet152,
                                                               len(test_loader_adv_generated.dataset),
                                                               100. * success_attack_resnet152 / len(
                                                                   test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'resnet152':
        transferability += 100. * success_attack_resnet152 / len(test_loader_adv_generated.dataset)
    # Inceptionv3
    success_attack_inceptionv3 = 0
    model_inceptionv3 = models.inception_v3()
    model_inceptionv3.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'inception_v3_checkpoint.pth')))
    model_inceptionv3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_inceptionv3
    )
    model_inceptionv3.to(device)
    model_inceptionv3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_inceptionv3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_inceptionv3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on Inceptionv3: ASR: {}/{} ({:.2f}%)'.format(success_attack_inceptionv3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_inceptionv3 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'inceptionv3':
        transferability += 100. * success_attack_inceptionv3 / len(test_loader_adv_generated.dataset)
    # MobileNetv2
    success_attack_mobilenetv2 = 0
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2 = nn.Sequential(
        Normalize(),
        model_mobilenetv2
    )
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on MobileNetv2: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_mobilenetv2 / len(
                                                                     test_loader_adv_generated.dataset)))
    ASR_list.append(100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset))
    if args.model_type != 'mobilenetv2':
        transferability += 100. * success_attack_mobilenetv2 / len(test_loader_adv_generated.dataset)

    print('Transferability:{:.2f}%'.format(transferability/5))
    print('time spended:{}'.format(time_spended))

    # The transferability to the defenses
    ASR_against_defense = []
    # 按照R&P, Bit-Red, JPEG, FD, NRP, RS, advInc-v3, ensAdvIncRes-v2的顺序进行评估
    # R&P
    rp = Randomization(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_rp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = rp.random_resize_pad(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_rp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on MobileNetv2 with RP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rp,
                                                                           len(test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_rp / len(
                                                                               test_loader_adv_generated.dataset)))
    # Bit-Red
    bit_red = BitDepthReduction(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_bit_red = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = bit_red.bit_depth_reduction(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_bit_red += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on MobileNetv2 with Bit-Red: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_bit_red,
                                                                                len(test_loader_adv_generated.dataset),
                                                                                100. * success_attack_mobilenetv2_bit_red / len(
                                                                                    test_loader_adv_generated.dataset)))
    # JPEG
    jpeg = Jpeg_compresssion(model_mobilenetv2, device, "ImageNet")
    success_attack_mobilenetv2_jpeg = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = jpeg.jpegcompression(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_jpeg += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on MobileNetv2 with JPEG: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_jpeg,
                                                                             len(
                                                                                 test_loader_adv_generated.dataset),
                                                                             100. * success_attack_mobilenetv2_jpeg / len(
                                                                                 test_loader_adv_generated.dataset)))
    # FD
    success_attack_mobilenetv2_fd = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv = data_adv.numpy()
        data_adv = FD_jpeg_encode(data_adv)
        data_adv = torch.from_numpy(data_adv)
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_mobilenetv2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_fd += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on MobileNetv2 with FD: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_fd,
                                                                           len(
                                                                               test_loader_adv_generated.dataset),
                                                                           100. * success_attack_mobilenetv2_fd / len(
                                                                               test_loader_adv_generated.dataset)))
    # NRP
    netG = NRP(3, 3, 64, 23)
    netG.load_state_dict(torch.load("./defense/saved_model/NRP.pth"))
    netG = netG.to(device)
    netG.eval()
    success_attack_mobilenetv2_nrp = 0
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        eps_ = 16 / 255
        data_adv_m = data_adv + torch.randn_like(data_adv) * 0.05
        data_adv_m = torch.min(torch.max(data_adv_m, data_adv - eps_), data_adv + eps_)
        data_adv_m = torch.clamp(data_adv_m, 0.0, 1.0)
        with torch.no_grad():
            purified = netG(data_adv_m).detach()
            output = model_mobilenetv2(purified)
        pred = output.max(1, keepdim=True)[1]
        success_attack_mobilenetv2_nrp += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on MobileNetv2 with NRP: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_nrp,
                                                                            len(
                                                                                test_loader_adv_generated.dataset),
                                                                            100. * success_attack_mobilenetv2_nrp / len(
                                                                                test_loader_adv_generated.dataset)))
    # RS
    model_mobilenetv2 = models.mobilenet_v2()
    model_mobilenetv2.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
    model_mobilenetv2.to(device)
    model_mobilenetv2.eval()
    rs = Smooth(model_mobilenetv2, 1000, 0.25)
    success_attack_mobilenetv2_rs = 0
    testset_adv_generated_RS = SelectedImagenet_adv_test_RS(data_adv_generated, target_adv)
    test_loader_adv_generated_RS = torch.utils.data.DataLoader(testset_adv_generated_RS, batch_size=1,
                                                               shuffle=False, pin_memory=True,
                                                               num_workers=0)
    for data_adv, target in test_loader_adv_generated_RS:
        data_adv = data_adv.to(device)
        output = rs.predict(data_adv, 100, 0.001, 100)
        success_attack_mobilenetv2_rs += int(output != target)
    print('Ada_DI_FOTA Test on MobileNetv2 with RS: ASR: {}/{} ({:.2f}%)'.format(success_attack_mobilenetv2_rs,
                                                                           len(
                                                                               test_loader_adv_generated_RS.dataset),
                                                                           100. * success_attack_mobilenetv2_rs / len(
                                                                               test_loader_adv_generated_RS.dataset)))
    # advInc-v3
    success_attack_advInc_v3 = 0
    model_advInc_v3 = timm.create_model('adv_inception_v3', pretrained=True)
    model_advInc_v3 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_advInc_v3
    )
    model_advInc_v3.to(device)
    model_advInc_v3.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_advInc_v3(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_advInc_v3 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on advInc-v3: ASR: {}/{} ({:.2f}%)'.format(success_attack_advInc_v3,
                                                                 len(test_loader_adv_generated.dataset),
                                                                 100. * success_attack_advInc_v3 / len(
                                                                     test_loader_adv_generated.dataset)))
    # ensAdvIncRes-v2
    success_attack_ensAdvIncRes_v2 = 0
    model_ensAdvIncRes_v2 = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
    model_ensAdvIncRes_v2 = nn.Sequential(
        Interpolate(torch.Size([299, 299]), 'bilinear'),
        Normalize(),
        model_ensAdvIncRes_v2
    )
    model_ensAdvIncRes_v2.to(device)
    model_ensAdvIncRes_v2.eval()
    for data_adv, target in test_loader_adv_generated:
        data_adv, target = data_adv.to(device), target.to(device)
        with torch.no_grad():
            output = model_ensAdvIncRes_v2(data_adv)
        pred = output.max(1, keepdim=True)[1]
        success_attack_ensAdvIncRes_v2 += (~(pred.eq(target.view_as(pred)))).sum().item()
    print('Ada_DI_FOTA Test on ensAdvIncRes-v2: ASR: {}/{} ({:.2f}%)'.format(success_attack_ensAdvIncRes_v2,
                                                                       len(test_loader_adv_generated.dataset),
                                                                       100. * success_attack_ensAdvIncRes_v2 / len(
                                                                           test_loader_adv_generated.dataset)))

    data_length = len(test_loader_adv_generated.dataset)
    data_length_rs = len(test_loader_adv_generated_RS.dataset)
    ASR_against_defense.append(
        [100. * success_attack_mobilenetv2_rp / data_length, 100. * success_attack_mobilenetv2_bit_red / data_length,
         100. * success_attack_mobilenetv2_jpeg / data_length,
         100. * success_attack_mobilenetv2_fd / data_length, 100. * success_attack_mobilenetv2_nrp / data_length,
         100. * success_attack_mobilenetv2_rs / data_length_rs,
         100. * success_attack_advInc_v3 / data_length, 100. * success_attack_ensAdvIncRes_v2 / data_length])
    print('Transferability to the defenses:{:.2f}%'.format(np.mean(ASR_against_defense)))

    return ASR_list, ASR_against_defense

def test_main():
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

    # 加载替代模型和受害者模型都分类正确的测试样本集
    transforms = T.Compose(
        [T.Resize(224), T.ToTensor()]
    )
    testset = ImageNet(args.input_dir, args.input_csv, transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True,
                                              num_workers=0)

    # 加载FuzzinessTuned方法需要的均值和方差
    correct_logit_mean_std = np.load(os.path.join(args.model_dir, args.model_type+"_correct_logit_mean_std_test.npy"))
    wrong_logit_mean_std = np.load(os.path.join(args.model_dir, args.model_type+"_wrong_logit_mean_std_test.npy"))
    correct_logit_mean_std = torch.from_numpy(correct_logit_mean_std)
    wrong_logit_mean_std = torch.from_numpy(wrong_logit_mean_std)
    correct_logit_mean_std = correct_logit_mean_std.to(device)
    wrong_logit_mean_std = wrong_logit_mean_std.to(device)

    ASR_matrix = {}

    ASR_list, ASR_against_defenses = Untarget_MIFGSM_test(model, device, test_loader, correct_logit_mean_std, wrong_logit_mean_std)
    ASR_matrix['MIFGSM'] = ASR_list
    ASR_matrix['MIFGSM_against_defenses'] = ASR_against_defenses

    ASR_list, ASR_against_defenses = Untarget_FOTA_test(model, device, test_loader, correct_logit_mean_std, wrong_logit_mean_std)
    ASR_matrix['FOTA'] = ASR_list
    ASR_matrix['FOTA_against_defenses'] = ASR_against_defenses

    ASR_list, ASR_against_defenses = Untarget_DI_MIFGSM_test(model, device, test_loader, correct_logit_mean_std, wrong_logit_mean_std)
    ASR_matrix['DI_MIFGSM'] = ASR_list
    ASR_matrix["DI_MIFGSM_against_defenses"] = ASR_against_defenses

    ASR_list, ASR_against_defenses = Untarget_DI_FOTA_test(model, device, test_loader, correct_logit_mean_std,
                                                             wrong_logit_mean_std)
    ASR_matrix['DI_FOTA'] = ASR_list
    ASR_matrix["DI_FOTA_against_defenses"] = ASR_against_defenses

    ASR_list, ASR_against_defenses = Untarget_Ada_DI_FOTA_test(model, device, test_loader, correct_logit_mean_std,
                                                             wrong_logit_mean_std)
    ASR_matrix['Ada_DI_FOTA'] = ASR_list
    ASR_matrix["Ada_DI_FOTA_against_defenses"] = ASR_against_defenses

    np.save(os.path.join(args.model_dir, args.model_type+'_evaluate_comparison_with_baselines.npy'), ASR_matrix)

if __name__ == '__main__':
    test_main()

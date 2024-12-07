import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal

from attack.attack import Attack
from attack.utils import temperature_scaling, fuzziness_tuned

class Ens_DI_FOTA(Attack):
    r"""

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """

    def __init__(self, model_set, eps=8/255, alpha=2/255, steps=5, decay=1.0, resize_rate=0.9, diversity_prob=0.5, tau=1.0, kappa1=1.0, kappa2=1.0, num_classes=100, correct_logit_mean_std_set=None, wrong_logit_mean_std_set=None):
        super().__init__("Ens_DI_FOTA", model_set[0])
        self.model_set = model_set
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self._supported_mode = ['default', 'targeted']
        self.num_classes = num_classes
        self.correct_logit_mean_std_set = correct_logit_mean_std_set
        self.wrong_logit_mean_std_set = wrong_logit_mean_std_set
        self.tau = tau
        self.kappa1 = kappa1
        self.kappa2 = kappa2

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, images, labels, lambda1=1.0, lambda2=0.5):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        lambda1 = torch.tensor(1-lambda1).to(self.device)
        lambda2 = torch.tensor(lambda2).to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)


        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        wrong_logit_mean_set = []
        wrong_logit_std_set = []
        correct_logit_mean_set = []
        correct_logit_std_set = []
        # 加载images的outputs中有关的隶属度函数
        for i in range(len(self.wrong_logit_mean_std_set)):
            wrong_logit_mean_set.append(torch.zeros(len(images), self.num_classes).to(self.device) + self.wrong_logit_mean_std_set[i][:, 0])
            wrong_logit_std_set.append(torch.zeros(len(images), self.num_classes).to(self.device) + self.wrong_logit_mean_std_set[i][:, 1])
            correct_logit_mean_set.append(torch.zeros(len(images), self.num_classes).to(self.device) + self.correct_logit_mean_std_set[i][:, 0])
            correct_logit_std_set.append(torch.zeros(len(images), self.num_classes).to(self.device) + self.correct_logit_mean_std_set[i][:, 1])

        for _ in range(self.steps):
            adv_images.requires_grad = True

            outputs_sum = torch.zeros(len(adv_images), self.num_classes).to(self.device)
            for i in range(len(self.model_set)):
                model = self.model_set[i]
                outputs = model(self.input_diversity(adv_images))
                wrong_logit_mean = wrong_logit_mean_set[i]
                wrong_logit_std = wrong_logit_std_set[i]
                correct_logit_mean = correct_logit_mean_set[i]
                correct_logit_std = correct_logit_std_set[i]

                # 确定成功攻击的对抗样本以及目标错误类别
                success_index = (outputs.argmax(dim=1) != labels)
                success_labels = outputs.argmax(dim=1)[success_index]
                # 确定outputs相同shape的bool矩阵
                truncate_index = torch.zeros_like(outputs)
                success_labels_onehot = F.one_hot(success_labels, self.num_classes)
                truncate_index[success_index] = success_labels_onehot.float()
                # 确定截断阈值
                membership_function_mean = wrong_logit_mean
                membership_function_std = wrong_logit_std
                membership_function_mean[truncate_index.bool()] = correct_logit_mean[truncate_index.bool()]
                membership_function_std[truncate_index.bool()] = correct_logit_std[truncate_index.bool()]
                GaussDistrib = Normal(membership_function_mean, membership_function_std)
                Thresholds2 = GaussDistrib.icdf(lambda2)

                truncate_index = (truncate_index.bool() & (outputs >= Thresholds2)).float()
                # 对outputs进行截断操作
                outputs[truncate_index.bool()] = Thresholds2[truncate_index.bool()]

                # 对原正确类别的outputs进行截断
                correct_labels_onehot = F.one_hot(labels, self.num_classes)
                Thresholds1 = GaussDistrib.icdf(lambda1)
                correct_truncate_index = (correct_labels_onehot.bool() & (outputs <= Thresholds1))
                outputs[correct_truncate_index] = Thresholds1[correct_truncate_index]

                # fuzziness tuned methods and temperature scaling
                outputs = fuzziness_tuned(outputs, labels, fuzzy_scale_true=self.kappa1, fuzzy_scale_wrong_pred=self.kappa2)
                outputs = temperature_scaling(outputs, temperature_scale=self.tau)

                outputs_sum += outputs

            outputs_avg = outputs_sum / len(self.model_set)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs_avg, target_labels)
            else:
                cost = loss(outputs_avg, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        if self._targeted:
            return adv_images, target_labels
        else:
            return adv_images

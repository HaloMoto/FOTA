import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal

from attack.attack import Attack
from attack.utils import temperature_scaling, fuzziness_tuned
from attack.helper import random_start_function, to_np_uint8

class Ada_DI_FOTA(Attack):
    r"""

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """

    def __init__(self, model, eps=8/255, alpha=2/255, max_steps=5, membership_degree_threshold=0.8, decay=1.0,
                 tau=1.0, kappa1=1.0, kappa2=1.0, num_classes=100,
                 correct_logit_mean_std=None, wrong_logit_mean_std=None):
        super().__init__("Ada_DI_FOTA", model)
        self.eps = eps
        self.max_steps = max_steps
        self.membership_degree_threshold = membership_degree_threshold
        self.decay = decay
        self.alpha = alpha
        self._supported_mode = ['default', 'targeted']
        self.num_classes = num_classes
        self.correct_logit_mean_std = correct_logit_mean_std
        self.wrong_logit_mean_std = wrong_logit_mean_std
        self.tau = tau
        self.kappa1 = kappa1
        self.kappa2 = kappa2

        self.resize_rate = 0.9
        self.diversity_prob = 0.5

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

    def forward(self, images, labels, lambda1=0.5, lambda2=1.0):
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

        # 加载images的outputs中有关的隶属度函数
        wrong_logit_mean = torch.zeros(len(images), self.num_classes).to(self.device) + self.wrong_logit_mean_std[:, 0]
        wrong_logit_std = torch.zeros(len(images), self.num_classes).to(self.device) + self.wrong_logit_mean_std[:, 1]
        correct_logit_mean = torch.zeros(len(images), self.num_classes).to(self.device) + self.correct_logit_mean_std[:, 0]
        correct_logit_std = torch.zeros(len(images), self.num_classes).to(self.device) + self.correct_logit_mean_std[:, 1]

        # All examples are initialized as True to indicate the attacking is not finished.
        finished_signs = torch.ones(len(images)).to(self.device)
        step_count = 0
        iteration_statistic = torch.zeros_like(finished_signs)

        while finished_signs.sum() != 0 and step_count < self.max_steps:
            membership_degrees = torch.zeros(len(images)).to(self.device)
            adv_images.requires_grad = True
            outputs = self.model(self.input_diversity(adv_images))

            # 确定成功攻击的对抗样本以及目标错误类别
            success_index = (outputs.argmax(dim=1) != labels)
            success_labels = outputs.argmax(dim=1)[success_index]
            # 确定outputs相同shape的bool矩阵
            truncate_index = torch.zeros_like(outputs)
            success_labels_onehot = F.one_hot(success_labels, self.num_classes)
            truncate_index[success_index] = success_labels_onehot.float()
            truncate_index_copied = truncate_index.clone()
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
            # calculate the membership degree of the true category
            membership_degree_matrix2 = GaussDistrib.cdf(outputs)
            membership_degrees[success_index] = (membership_degree_matrix2[truncate_index_copied.bool()]) / lambda2

            # 对原正确类别的outputs进行截断
            correct_labels_onehot = F.one_hot(labels, self.num_classes)
            Thresholds1 = GaussDistrib.icdf(lambda1)
            correct_truncate_index = (correct_labels_onehot.bool() & (outputs <= Thresholds1))
            outputs[correct_truncate_index] = Thresholds1[correct_truncate_index]
            # calculate the membership degree of the misclassified category
            membership_degree_matrix1 = GaussDistrib.cdf(outputs)
            membership_degrees += (1-membership_degree_matrix1[correct_labels_onehot.bool()]) / (1-lambda1)
            # achieve the final degree
            membership_degrees = membership_degrees / 2

            # update the finished sign, 0 indicates finished, 1 indicates un-finished
            finished_signs = ((finished_signs.bool()) & (membership_degrees<self.membership_degree_threshold)).detach().float()
            # update the iteration steps of new finished images
            iteration_statistic[(~finished_signs.bool())&(iteration_statistic==0)] = step_count
            step_count += 1

            # fuzziness tuned methods and temperature scaling
            outputs = fuzziness_tuned(outputs, labels, fuzzy_scale_true=self.kappa1, fuzzy_scale_wrong_pred=self.kappa2)
            outputs = temperature_scaling(outputs, temperature_scale=self.tau)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            # update the unfinished images
            adv_images = adv_images.detach()
            adv_images_temp = adv_images + self.alpha*grad.sign()
            adv_images[finished_signs.bool()] = (adv_images_temp.detach())[finished_signs.bool()]
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        iteration_statistic[iteration_statistic==0] = self.max_steps

        if self._targeted:
            return adv_images, target_labels
        else:
            return adv_images, iteration_statistic.detach().cpu().tolist(), membership_degrees.detach().cpu().tolist()

    def forward_for_RR(self, images, labels, restart, lambda1=0.5, lambda2=1.0):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        lambda1 = torch.tensor(1-lambda1).to(self.device)
        lambda2 = torch.tensor(lambda2).to(self.device)
        img_ls, loss_ls = [], []

        # 加载images的outputs中有关的隶属度函数
        wrong_logit_mean = torch.zeros(len(images), self.num_classes).to(self.device) + self.wrong_logit_mean_std[:, 0]
        wrong_logit_std = torch.zeros(len(images), self.num_classes).to(self.device) + self.wrong_logit_mean_std[:, 1]
        correct_logit_mean = torch.zeros(len(images), self.num_classes).to(self.device) + self.correct_logit_mean_std[:, 0]
        correct_logit_std = torch.zeros(len(images), self.num_classes).to(self.device) + self.correct_logit_mean_std[:, 1]

        for restart_ind in range(max(restart, 1)):
            img_re_ls, loss_re_ls = [], []
            momentum = torch.zeros_like(images).detach().to(self.device)

            loss = nn.CrossEntropyLoss(reduction="none")

            if restart > 0:
                adv_images = random_start_function(images, self.eps)
            elif restart == 0:
                adv_images = images.clone().detach()
            else:
                raise ValueError("invalid argument restart.")

            # All examples are initialized as True to indicate the attacking is not finished.
            finished_signs = torch.ones(len(images)).to(self.device)
            step_count = 0

            while finished_signs.sum() != 0 and step_count < self.max_steps:
                membership_degrees = torch.zeros(len(images)).to(self.device)
                adv_images.requires_grad = True
                outputs = self.model(self.input_diversity(adv_images))

                # 确定成功攻击的对抗样本以及目标错误类别
                success_index = (outputs.argmax(dim=1) != labels)
                success_labels = outputs.argmax(dim=1)[success_index]
                # 确定outputs相同shape的bool矩阵
                truncate_index = torch.zeros_like(outputs)
                success_labels_onehot = F.one_hot(success_labels, self.num_classes)
                truncate_index[success_index] = success_labels_onehot.float()
                truncate_index_copied = truncate_index.clone()
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
                # calculate the membership degree of the true category
                membership_degree_matrix2 = GaussDistrib.cdf(outputs)
                membership_degrees[success_index] = (membership_degree_matrix2[truncate_index_copied.bool()]) / lambda2

                # 对原正确类别的outputs进行截断
                correct_labels_onehot = F.one_hot(labels, self.num_classes)
                Thresholds1 = GaussDistrib.icdf(lambda1)
                correct_truncate_index = (correct_labels_onehot.bool() & (outputs <= Thresholds1))
                outputs[correct_truncate_index] = Thresholds1[correct_truncate_index]
                # calculate the membership degree of the misclassified category
                membership_degree_matrix1 = GaussDistrib.cdf(outputs)
                membership_degrees += (1-membership_degree_matrix1[correct_labels_onehot.bool()]) / (1-lambda1)
                # achieve the final degree
                membership_degrees = membership_degrees / 2

                # update the finished sign, 0 indicates finished, 1 indicates un-finished
                finished_signs = ((finished_signs.bool()) & (membership_degrees<self.membership_degree_threshold)).detach().float()
                step_count += 1

                # fuzziness tuned methods and temperature scaling
                outputs = fuzziness_tuned(outputs, labels, fuzzy_scale_true=self.kappa1, fuzzy_scale_wrong_pred=self.kappa2)
                outputs = temperature_scaling(outputs, temperature_scale=self.tau)

                # Calculate loss
                cost_ = loss(outputs, labels)
                cost = cost_.mean()

                img_re_ls.append(to_np_uint8(adv_images.data.clone()))
                loss_re_ls.append(cost_.view(-1).data.clone())

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
                grad = grad + momentum*self.decay
                momentum = grad

                # update the unfinished images
                adv_images = adv_images.detach()
                adv_images_temp = adv_images + self.alpha*grad.sign()
                adv_images[finished_signs.bool()] = (adv_images_temp.detach())[finished_signs.bool()]
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            img_ls.append(np.stack(img_re_ls))
            loss_ls.append(torch.stack(loss_re_ls).cpu().numpy())
        img_ls = np.stack(img_ls)
        loss_ls = np.stack(loss_ls)

        return img_ls, loss_ls

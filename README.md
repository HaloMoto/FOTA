# Rethinking the Optimization Objective for Transferable Adversarial Examples from a Fuzzy Perspective (Accpeted by Neural Networks 2024)

This repository is the official Pytorch code implementation for our paper "Rethinking the Optimization Objective for Transferable Adversarial Examples from a Fuzzy Perspective". In this paper, we propose a fuzzy optimization-based transferable attack (FOTA) to maximize both the original cross-entropy loss and the newly proposed membership functions. The proposed membership functions are positively correlated to the transferability of adversarial examples.

## Requirements

<ul>
<li>Python=3.7</li>
<li>Torch=1.10</li>
<li>Torchvision=0.11</li>
<li>Numpy=1.18</li>
<li>Pandas=1.0</li>
<li>Scipy=1.4</li>
</ul>

## Implementation

- Download the models (Place the pretrained weights in the "saved_models" folder)

  - [Naturally trained models](https://pytorch.org/vision/stable/models.html)

  - [Adversarially trained models](https://huggingface.co/models)

- Place these pretrained weights in the "saved_models" folder

- Run the code

  - Single surrogate model: choose one of surrogate model from the set \{VGG16, ResNet50, ResNet152\}

    ```python
    python evaluate_comparison_with_baselines.py --model-type vgg16

    python evaluate_comparison_with_baselines.py --model-type resnet50

    python evaluate_comparison_with_baselines.py --model-type resnet152
    ```

  - Ensemble surrogate model: is consisted by \{VGG16, ResNet50, ResNet152\}
   
    ```python
    python evaluate_comparison_with_baselines_ens.py --model-type Ens
    ```

  - If you want to choose other surrogate models such as MobileNetv2, you need to firstly run the following code to achieve the distribution of manifold feature.
 
    ```python
    python calculate_mean_std_of_logit.py --model-type mobilenetv2
    ```

## Acknowledgments

Code refers to [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) and [VT](https://github.com/JHL-HUST/VT)

## Citing this work

If you find this work is useful in your research, please consider citing:

```
xxx
```

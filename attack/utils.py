import torch
import torch.nn.functional as F

def temperature_scaling(outputs, temperature_scale=1.0):
    outputs = outputs / temperature_scale
    return outputs

def fuzziness_tuned(outputs, ys, fuzzy_scale_true=1.0, fuzzy_scale_wrong_pred=1.0):
    # fuzziness tuned y_true
    ys_onehot = F.one_hot(ys, outputs.shape[1])
    outputs[ys_onehot.bool()] = outputs[ys_onehot.bool()] * fuzzy_scale_true
    # fuzziness tuned maximum wrong category
    '''
    outputs_copied = outputs.clone()
    outputs_copied[ys_onehot.bool()] = -torch.inf
    wrong_categories = outputs_copied.argmax(dim=1)
    wrong_categories_onehot = F.one_hot(wrong_categories, outputs.shape[1])
    outputs[wrong_categories_onehot.bool()] = outputs[wrong_categories_onehot.bool()] * fuzzy_scale_wrong_pred
    '''
    success_index = (outputs.argmax(dim=1) != ys)
    success_labels = outputs.argmax(dim=1)[success_index]
    wrong_pred_index = torch.zeros_like(outputs)
    success_labels_onehot = F.one_hot(success_labels, outputs.shape[1])
    wrong_pred_index[success_index] = success_labels_onehot.float()
    outputs[wrong_pred_index.bool()] = outputs[wrong_pred_index.bool()] * fuzzy_scale_wrong_pred

    return outputs
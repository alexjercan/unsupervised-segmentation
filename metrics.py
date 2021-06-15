# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://github.com/XinJCheng/CSPN/blob/b3e487bdcdcd8a63333656e69b3268698e543181/cspn_pytorch/utils.py#L19
# - https://web.eecs.umich.edu/~fouhey/2016/evalSN/evalSN.html
#


from general import generate_layers, generate_surfaces, layers_to_canvas
import torch


class MetricFunction():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.total_size = 0
        self.error_sum = {}
        self.error_avg = {}

    def evaluate(self, predictions, data):
        (normals, depths) = data
        num_layers = predictions.shape[-1]
        _, predictions = torch.max(predictions, 1)

        surfaces = generate_surfaces(normals)
        layers = generate_layers(surfaces, depths, num_layers)
        layers = torch.stack(layers, dim=-1).squeeze(1) * predictions

        canvas = layers_to_canvas(predictions)
        gt_canvas = layers_to_canvas(layers)

        error_val = evaluate_error_classification(canvas, gt_canvas)

        self.total_size += self.batch_size
        self.error_avg = avg_error(self.error_sum, error_val, self.total_size, self.batch_size)
        return self.error_avg

    def show(self):
        error = self.error_avg
        format_str = ('======SEGMENTATION========\nIOU=%.4f\tP=%.4f\tR=%.4f\tF1=%.4f\n')
        return format_str % (error['S_IOU'], error['S_P'], error['S_R'], error['S_F1'])


class MetricFunctionNYUv2():
    def __init__(self, batch_size) -> None:
        self.batch_size = batch_size
        self.total_size = 0
        self.error_sum = {}
        self.error_avg = {}

    def evaluate(self, predictions, data):
        (seg13, depths) = data
        num_layers = predictions.shape[-1]
        _, predictions = torch.max(predictions, 1)

        error_val = evaluate_error_classification(predictions, seg13)

        self.total_size += self.batch_size
        self.error_avg = avg_error(self.error_sum, error_val, self.total_size, self.batch_size)
        return self.error_avg

    def show(self):
        error = self.error_avg
        format_str = ('======SEGMENTATION========\nIOU=%.4f\tP=%.4f\tR=%.4f\tF1=%.4f\n')
        return format_str % (error['S_IOU'], error['S_P'], error['S_R'], error['S_F1'])



def evaluate_error_classification(predictions, targets):
    error = {}
    p_mask = (predictions > 0).float()
    gt_mask = (targets > 0).float()

    intersection = torch.sum(p_mask * gt_mask)
    union = torch.sum(torch.clamp(p_mask + gt_mask, 0, 1))

    tp = torch.logical_and(predictions > 0, predictions == targets).float().sum()
    fp = torch.logical_and(predictions > 0, targets == 0).float().sum()
    fn = torch.logical_and(predictions == 0, targets > 0).float().sum()

    error['S_IOU'] = intersection / union
    error['S_P'] = tp / (tp + fp)
    error['S_R'] = tp / (tp + fn)
    error['S_F1'] = 2 * (error['S_P'] * error['S_R'] / (error['S_P'] + error['S_R']))
    return error


# avg the error
def avg_error(error_sum, error_val, total_size, batch_size):
    error_avg = {}
    for item, value in error_val.items():
        error_sum[item] = error_sum.get(item, 0) + value * batch_size
        error_avg[item] = error_sum[item] / float(total_size)
    return error_avg


def print_single_error(epoch, loss, error):
    format_str = ('%s\nEpoch: %d, loss=%s\n%s\n')
    print (format_str % ('eval_avg_error', epoch, loss, error))
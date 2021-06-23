# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://github.com/XinJCheng/CSPN/blob/b3e487bdcdcd8a63333656e69b3268698e543181/cspn_pytorch/utils.py#L19
# - https://web.eecs.umich.edu/~fouhey/2016/evalSN/evalSN.html
#


from general import generate_layers, generate_surfaces, layers_to_canvas, squash_layers
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

        predictions = torch.clamp(predictions, 0, 1)
        layers = torch.clamp(layers, 0, 1)

        error_val = evaluate_error_classification(predictions, layers)

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
        (seg13, normals, depths) = data
        num_layers = predictions.shape[-1]
        _, predictions = torch.max(predictions, 1)

        device = predictions.device
        canvas = torch.zeros(predictions.shape[:-1], dtype=torch.long, device=device)
        predictions = torch.stack(squash_layers(predictions, depths, predictions.shape[-1]))
        for pred in predictions.permute(1, 0, 2, 3):
            canvas = torch.where(pred != -1, pred, canvas)

        canvas_p = torch.clamp(canvas, 0, 1)
        seg13_p = torch.clamp(seg13, 0, 1)

        error_val = evaluate_error_classification(canvas_p.unsqueeze(1), seg13_p)

        self.total_size += self.batch_size
        self.error_avg = avg_error(self.error_sum, error_val, self.total_size, self.batch_size)
        return self.error_avg

    def show(self):
        error = self.error_avg
        format_str = ('======SEGMENTATION========\nIOU=%.4f\tP=%.4f\tR=%.4f\tF1=%.4f\n')
        return format_str % (error['S_IOU'], error['S_P'], error['S_R'], error['S_F1'])



def evaluate_error_classification(predictions, targets):
    error = {}

    intersection = torch.sum(predictions * targets)
    union = torch.sum(torch.clamp(predictions + targets, 0, 1))

    intersection0 = torch.logical_and(predictions == 0, targets == 0).float().sum()
    union0 = torch.logical_or(predictions == 0, targets == 0).float().sum()

    a = intersection / union if union != 0 else 0
    b = intersection0 / union0 if union0 != 0 else 0

    tp = torch.logical_and(predictions > 0, predictions == targets).float().sum()
    fp = torch.logical_and(predictions > 0, targets == 0).float().sum()
    fn = torch.logical_and(predictions == 0, targets > 0).float().sum()

    error['S_IOU'] = (a + b) / 2
    error['S_P'] = tp / (tp + fp) if (tp + fp) != 0 else 0
    error['S_R'] = tp / (tp + fn) if (tp + fn) != 0 else 0
    error['S_F1'] = 2 * (error['S_P'] * error['S_R'] / (error['S_P'] + error['S_R'])) if (error['S_P'] + error['S_R']) != 0 else 0
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
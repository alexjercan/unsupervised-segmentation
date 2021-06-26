import os
import re
from datetime import datetime as dt
from metrics import MetricFunctionNYUv2, print_single_error
from general import generate_layers, save_checkpoint, set_parameter_requires_grad, tensors_to_device
from model import SupervisedLossFunction
from torchvision.models.segmentation import fcn_resnet50
from torchvision import transforms
from torchvision.models.segmentation.fcn import FCNHead
from nyuv2 import NYUv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

num_layers = 3


def runmodel(model, imgs, depths):
    layers = generate_layers(imgs, depths, num_layers)
    # plot_raw_surfaces(imgs.permute(0, 2, 3, 1), torch.stack(layers, dim=-1))
    x = [model(x)['out'] for x in layers]
    return torch.stack(x, dim=-1)


def train_one_epoch_nyuv2(model, dataloader, loss_fn, metric_fn, solver, epoch_index):
    loop = tqdm(dataloader, position=0, leave=True)

    for i, tensors in enumerate(loop):
        imgs, seg13, normals, depths = tensors_to_device(tensors, DEVICE)

        predictions = runmodel(model, imgs, depths)

        loss = loss_fn(predictions, (seg13, depths))
        metric_fn.evaluate(predictions, (seg13, normals, depths))

        model.zero_grad()
        loss.backward()
        solver.step()

        loop.set_postfix(loss=loss_fn.show(), epoch=epoch_index)
    loop.close()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = fcn_resnet50(pretrained=True, num_classes=21)
set_parameter_requires_grad(model)
model.classifier = FCNHead(2048, channels=14)
model = model.to(DEVICE)

loss_fn = SupervisedLossFunction()

solver = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver, milestones=[10], gamma=0.1)

t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
train_dataset = NYUv2(root="../NYUv2", download=True, rgb_transform=t, seg_transform=t, sn_transform=t, depth_transform=t, train=True)
dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

output_dir = os.path.join("./runs", re.sub("[^0-9a-zA-Z]+", "-", dt.now().isoformat()))
for epoch_idx in range(0, 5):
    metric_fn = MetricFunctionNYUv2(2)

    model.train()
    train_one_epoch_nyuv2(model, dataloader, loss_fn, metric_fn, solver, epoch_idx)
    print_single_error(epoch_idx, loss_fn.show(), metric_fn.show())
    lr_scheduler.step()
    save_checkpoint(epoch_idx, model, output_dir)

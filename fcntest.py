from metrics import MetricFunctionNYUv2, print_single_error
from model import SupervisedLossFunction
from torch.utils.data import DataLoader
from torchvision import transforms
from nyuv2 import NYUv2
from tqdm import tqdm
from general import generate_layers, load_checkpoint, tensors_to_device
import torch
from torchvision.models.segmentation.segmentation import fcn_resnet50

num_layers = 3


def runmodel(model, imgs, depths):
    layers = generate_layers(imgs, depths, num_layers)
    x = [model(x)['out'] for x in layers]
    return torch.stack(x, dim=-1)


def run_test_nyuv2(model, dataloader, loss_fn, metric_fn):
    loop = tqdm(dataloader, position=0, leave=True)

    for i, tensors in enumerate(loop):
        imgs, seg13, normals, depths = tensors_to_device(tensors, DEVICE)
        with torch.no_grad():
            predictions = runmodel(model, imgs, depths)

            loss_fn(predictions, (normals, depths))
            metric_fn.evaluate(predictions, (seg13, normals, depths))
    loop.close()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = fcn_resnet50(pretrained=False, num_classes=14)
model = model.to(DEVICE)
epoch_idx, model = load_checkpoint(model, "fcnmodel.pth", DEVICE)

t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
test_dataset = NYUv2(root="../NYUv2", download=True, rgb_transform=t, seg_transform=t, sn_transform=t, depth_transform=t, train=False)
dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

loss_fn = SupervisedLossFunction()
metric_fn = MetricFunctionNYUv2(2)

model.eval()
run_test_nyuv2(model, dataloader, loss_fn, metric_fn)
print_single_error(epoch_idx, loss_fn.show(), metric_fn.show())
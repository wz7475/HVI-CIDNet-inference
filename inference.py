import os.path
from argparse import ArgumentParser
from typing import List
from uuid import uuid4

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image

from net.CIDNet import CIDNet


def infer_transforms(size=640):
    return Compose([
        Resize((size, size)),
        ToTensor(),
    ])


class InferenceDataset(Dataset):
    def __init__(self, data_dir: str, transform=infer_transforms()):
        self.transform = transform
        self.img_names = [filename for filename in os.listdir(data_dir) if filename.split(".")[-1] in ['jpg', 'jpeg', 'png', 'ppm', 'JPG']]
        self.img_paths = [os.path.join(data_dir, filename) for filename in self.img_names]

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("RGB")
        print(self.img_paths[index])
        return self.transform(img), self.img_names[index]

    def __len__(self):
        return len(self.img_paths)


def save_batch(batch: torch.Tensor, out_dir: str, img_names: List[str]) -> None:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for idx, tensor_3d in enumerate(batch):
        img = to_pil_image(tensor_3d)
        img.save(os.path.join(out_dir, img_names[idx]))


def inference_for_dir(dir_in: str, dir_out: str, model: torch.nn.Module, batch_size: int = 1):
    dataset = InferenceDataset(data_dir=dir_in)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for idx, batch in enumerate(dataloader):
        print(f"{idx}/{len(dataloader)}")
        images, names = batch
        output = model(images.to("cuda"))
        save_batch(output, dir_out, names)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir_in", help="dir with images")
    parser.add_argument("dir_out", help="dir with images")
    args = parser.parse_args()
    dir_in = args.dir_in
    dir_out = args.dir_out
    # model
    model = CIDNet()
    model.load_state_dict(torch.load("weights/best_SSIM.pth"))
    model.to("cuda")
    model.eval()

    # inference
    inference_for_dir(dir_in, dir_out, model, batch_size=1)
    print("done")

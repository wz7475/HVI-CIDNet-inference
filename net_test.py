import os.path
from argparse import ArgumentParser
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
        self.img_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        return self.transform(img)

    def __len__(self):
        return len(self.img_paths)

def save_batch(batch: torch.Tensor, out_dir: str) -> None:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for tensor_3d in batch:
        img = to_pil_image(tensor_3d)
        img.save(os.path.join(out_dir, f"{uuid4()}.jpg"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dir", help="dir with images")
    directory = parser.parse_args().dir
    # model
    model = CIDNet()
    model.load_state_dict(torch.load("weights/best_SSIM.pth"))
    model.to("cuda")
    model.eval()

    # dataset
    dataset = InferenceDataset(directory)
    dataloader = DataLoader(dataset, batch_size=1)

    # inference
    torch.cuda.empty_cache()
    for batch in dataloader:
        output = model(batch.to("cuda"))
        save_batch(output, "images_out")

    print("done")

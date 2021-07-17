from torch.utils.data.dataset import Dataset
import os
import natsort
from PIL import Image


class Data(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_images = os.listdir(main_dir)
        self.total_imges = natsort.natsorted(all_images)

    def __len__(self):
        return len(self.total_imges)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imges[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

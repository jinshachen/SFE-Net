import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from transforms import pad_if_smaller


class CellDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None, Norm=None):
        super(CellDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        self.train = train
        self.Norm = Norm
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith((".tif", ".bmp", ".png"))]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.mask_list = [os.path.join(data_root, "masks", i.split(".")[0] + ".png") for i in img_names]
        # check files
        for i in self.mask_list:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(self.img_list[idx]).convert('RGB')
        crop_size = 256
        img = pad_if_smaller(img, crop_size, fill=0)

        mask = Image.open(self.mask_list[idx]).convert('L')
        mask = pad_if_smaller(mask, crop_size, fill=0)
        mask = np.array(mask)
        mask[mask != 0] = 1

        mask = Image.fromarray(mask)

        if self.Norm is not None:
            if self.train:
                img, mask = self.transforms(img, mask)
                img, mask = self.Norm(img, mask)
            else:
                img, mask = self.Norm(img, mask)
            return img, mask, img_path
        return img, mask, img_path

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets, img_paths = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets, img_paths


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

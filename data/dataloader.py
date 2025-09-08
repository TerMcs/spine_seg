import cv2
from torch.utils import data


class SegmentationDataset(data.Dataset):
    def __init__(self, split, trf):
        self.split = split
        self.transforms = trf

    def __getitem__(self, idx):
        entry = self.split.iloc[idx]
        img_fname = entry.img_fname
        mask_fname = entry.mask_fname

        img = self.read_img(img_fname)
        mask = self.read_mask(mask_fname)

        res = self.transforms({'image': img, 'mask': mask})

        res = {'img': res['image'], 'mask': res['mask'].long(), 'fname': img_fname.name}

        return res

    @staticmethod
    def read_img(path):
        return cv2.imread(str(path))

    @staticmethod
    def read_mask(path):
        return cv2.imread(str(path))[:, :, 0]

    def __len__(self):
        return self.split.shape[0]

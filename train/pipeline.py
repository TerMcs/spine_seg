import cv2

import pathlib
import torch
import numpy as np
import pandas as pd
import hydra

import time
import socket
import logging

from sklearn.model_selection import GroupKFold
from data.dataloader import SegmentationDataset
from data.get_masks import LABEL_MAPPING
from utils.train import save_checkpoint, pass_epoch, init_augs, fix_seed
from torch.utils.data import DataLoader, SequentialSampler

from segmentation_models_pytorch.losses import JaccardLoss

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='/mnt/sda1/lseg/outputs/runs')

INV_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

log = logging.getLogger(__name__)

class CombinedLoss(torch.nn.Module):
    def __init__(self, log_jaccard=False):
        super(CombinedLoss, self).__init__()

        self.jaccard = JaccardLoss(mode='multiclass', log_loss=log_jaccard)
        self.CE = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        return self.CE(x, y) + self.jaccard(x, y)


def train_model(cfg: DictConfig):
    data_dir = pathlib.Path(cfg.data_dir)
    work_dir = pathlib.Path(cfg.work_dir)

    fix_seed(cfg.seed)

    snapshot_dir = work_dir / 'snapshots' / time.strftime(f'{socket.gethostname()}_%Y_%m_%d_%H_%M_%S')
    snapshot_dir.mkdir(parents=True)

    df = []
    for fname in data_dir.glob('images_21-11-10a/*'):
        df.append({'subject_id': fname.name.split('_IMG')[0],
                    'img_fname': fname,
                    'mask_fname': data_dir / 'masks_21-11-10a' / fname.name})

    df = pd.DataFrame(data=df)
    splits = GroupKFold(n_splits=cfg.n_folds).split(df, groups=df.subject_id)

    train_augs, val_augs = init_augs()
    for fold_id, (train_idx, val_idx) in enumerate(splits):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_ds = SegmentationDataset(train_df, train_augs)
        val_ds = SegmentationDataset(val_df, val_augs)

        val_sampler = SequentialSampler(val_ds)
        train_loader = DataLoader(dataset=train_ds, batch_size=cfg.bs, shuffle=True, num_workers=cfg.n_workers)
        val_loader = DataLoader(dataset=val_ds, batch_size=cfg.bs, num_workers=cfg.n_workers, sampler=val_sampler)

        model = hydra.utils.instantiate(cfg.model).to('cuda')

        criterion = CombinedLoss(log_jaccard=cfg.log_jaccard)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=cfg.wd)

        best_snapshot_path = None
        best_val_metric = None

        for epoch in range(cfg.max_epochs):
            train_loss, _ = pass_epoch(fold_id, epoch, cfg.max_epochs, cfg.model.classes, model,
                                       train_loader, optimizer, criterion)
            val_loss, iou_list = pass_epoch(fold_id, epoch, cfg.max_epochs, cfg.model.classes, model,
                                            val_loader, None, criterion)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)

            log.info(f'[Epoch {epoch} ~ Fold [{fold_id}] | Train loss {train_loss:.4f}, Val. loss {val_loss:.4f}')

            for i in range(iou_list.shape[0]):
                log.info(f'{INV_LABEL_MAPPING[i]}: {iou_list[i]:.4f}')

            iou_metric = np.mean(iou_list[1:])

            writer.add_scalar("Avg. IOU metric", iou_metric, epoch)

            best_snapshot_path, best_val_metric = save_checkpoint(fold_id, epoch, snapshot_dir,
                                                                  best_snapshot_path, iou_metric,
                                                                  best_val_metric, model, 'gt')

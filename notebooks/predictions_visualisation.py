import torch
import tqdm
import cv2
import pathlib
import hydra

import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import gc

from omegaconf import DictConfig, OmegaConf

from utils.train import init_augs
from data.get_masks import LABEL_MAPPING

INV_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


WORKDIR = pathlib.Path('outputs')
ROOT_DIR = pathlib.Path('/mnt/sda1/lseg/outputs/snapshots/tmcs_2021_11_10_12_45_28')
METADATA_PATH = ROOT_DIR / 'worst_predictions_full_paths.csv'
CONFIG_PATH = WORKDIR / '2021-11-19/11-02-50/.hydra'
SNAPSHOT_PATH = WORKDIR / 'snapshots/tmcs_2021_11_19_11_02_50'
with open(CONFIG_PATH / 'config.yaml', 'rb') as f:
    scfg = OmegaConf.load(f)

PICS = SNAPSHOT_PATH / 'new_inference_on_worst_predictions'
if PICS.is_dir():
    shutil.rmtree(PICS)

PICS.mkdir()

models = []

for fold_id in range(scfg.n_folds):
    model = hydra.utils.instantiate(scfg.model).to('cuda')

    snp_name = next(SNAPSHOT_PATH.glob(f'fold_{fold_id}_*.pth'))

    state = torch.load(snp_name)['model']
    model.load_state_dict(state)
    model = model.to('cuda')
    model.eval()
    models.append(model)

_, solt_trf = init_augs()

gc.collect()

meta = pd.read_csv(METADATA_PATH, index_col=0)
for row_id, entry in meta.iterrows():
    if row_id == 0:
        continue
    png_fname = entry.FullPath
    img = cv2.imread(str(png_fname))
    img_torch = solt_trf(img)['image'].unsqueeze(0).to('cuda')
    pred = 0
    with torch.no_grad():
        for m in models:
            pred += m(img_torch)
        pred /= len(models)
    pred = pred.squeeze().argmax(0).to('cpu').numpy()

    plt.figure(figsize=(30, 10))
    plt.subplot(141)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(142)
    plt.imshow(pred)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(143)
    img_contours = img.copy()
    for cls_id in range(1, scfg.model.classes):
        cur_mask = np.uint8(pred == cls_id) * 255
        if cur_mask.sum() == 0:
            continue
        contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cls_id % 2:
            cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    plt.imshow(img_contours)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(144)
    img_contours = img.copy()
    for cls_id in range(1, scfg.model.classes):
        cur_mask = np.uint8(pred == cls_id) * 255
        if cur_mask.sum() == 0:
            continue
        contours, hierarchy = cv2.findContours(cur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if cls_id % 2 == 0:
            cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 3)

    plt.imshow(img_contours)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(PICS / f'{entry.image_ID}', bbox_inches='tight')
    plt.close()

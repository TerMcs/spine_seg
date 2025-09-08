import torch
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

from tqdm import tqdm

INV_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


WORKDIR = pathlib.Path('outputs')
ROOT_DIR = pathlib.Path('/mnt/sda1/lseg')
IMAGE_DIR = pathlib.Path('/mnt/sda1/nfbc')
METADATA_PATH = ROOT_DIR / 'outputs/snapshots/tmcs_2021_11_19_11_02_50/best_predictions.csv'
CONFIG_PATH = WORKDIR / '2021-11-19/11-02-50/.hydra'
SNAPSHOT_PATH = WORKDIR / 'snapshots/tmcs_2021_11_19_11_02_50'
with open(CONFIG_PATH / 'config.yaml', 'rb') as f:
    scfg = OmegaConf.load(f)

MASKS = WORKDIR / 'snapshots/tmcs_2021_11_19_11_02_50/best_predictions'
if MASKS.is_dir():
    shutil.rmtree(MASKS)

MASKS.mkdir()

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
    # if row_id == 0:
    #     continue
    png_fname = IMAGE_DIR / (entry.FullPathDICOM[1:] + '.png')
    img = cv2.imread(str(png_fname))
    img_torch = solt_trf(img)['image'].unsqueeze(0).to('cuda')
    pred = 0
    with torch.no_grad():
        for m in models:
            pred += m(img_torch)
        pred /= len(models)
    pred = pred.squeeze().argmax(0).to('cpu').numpy()

    plt.imshow(pred)
    plt.xticks([])
    plt.yticks([])

    plt.imsave(MASKS / f'{entry.image_ID}', pred)
    plt.close()
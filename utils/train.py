import solt
import solt.transforms as slt
import numpy as np
import gc
import torch
import operator
import random
import logging
from tqdm import tqdm
from .eval import calculate_iou, calculate_confusion_matrix_from_arrays

log = logging.getLogger(__name__)


def pass_epoch(fold_id, epoch, max_epoch, n_classes, net, loader, optimizer, criterion):
    net.train(optimizer is not None)

    running_loss = 0.0
    n_batches = len(loader)

    device = next(net.parameters()).device
    pbar = tqdm(total=n_batches, ncols=200)

    iou_list = []
    with torch.set_grad_enabled(optimizer is not None):
        for i, entry in enumerate(loader):
            if optimizer is not None:
                optimizer.zero_grad()

            inputs = entry['img'].to(device)
            mask = entry['mask'].to(device).squeeze()
            outputs = net(inputs)
            loss = criterion(outputs, mask)

            if optimizer is not None:
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_description(f"Fold [{fold_id}] [{epoch} | {max_epoch}] | "
                                     f"Running loss {running_loss / (i + 1):.5f} / {loss.item():.5f}")
            else:
                running_loss += loss.item()
                pbar.set_description(desc=f"Fold [{fold_id}] [{epoch} | {max_epoch}] | Validation progress")

                preds = outputs.argmax(axis=1)

                preds = preds.float().to('cpu').numpy()
                mask = mask.float().to('cpu').numpy()
                for batch_el_id in range(preds.shape[0]):
                    confusion_matrix = calculate_confusion_matrix_from_arrays(preds[batch_el_id, :, :],
                                                                              mask[batch_el_id, :, :], n_classes)
                    iou_res = calculate_iou(confusion_matrix)
                    iou_list.append(np.array(iou_res))
            pbar.update()
            gc.collect()
        gc.collect()
        pbar.close()

    if optimizer is None:
        iou_list = np.mean(iou_list, axis=0)

    return running_loss / n_batches, iou_list


def save_checkpoint(fold_id, epoch,
                    snapshot_dir_path, prev_snapshot_path,
                    val_metric, best_val_metric, net, comparator='gt'):

    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    comparator = getattr(operator, comparator)
    metric_str = f'{val_metric:.4}'.replace('.', '__')
    cur_snapshot_path = snapshot_dir_path / f'fold_{fold_id}_epoch_{epoch}_{metric_str}.pth'

    state = {'model': net.state_dict()}
    if best_val_metric is None:
        print('====> Snapshot was saved to', cur_snapshot_path)
        torch.save(state, cur_snapshot_path)
        prev_snapshot_path = cur_snapshot_path
        best_val_metric = val_metric
        log.info('Snapshot saved.')

    else:
        if comparator(val_metric, best_val_metric):
            print('====> Snapshot was saved to', cur_snapshot_path)
            prev_snapshot_path.unlink()
            torch.save(state, cur_snapshot_path)
            prev_snapshot_path = cur_snapshot_path
            best_val_metric = val_metric
            log.info('Snapshot saved.')

    return prev_snapshot_path, best_val_metric


def init_augs():
    train_augs = solt.Stream([
        slt.Pad((540, 540)),
        slt.Crop((480, 480), crop_mode='r'),
        slt.GammaCorrection(gamma_range=0.5, p=1),
    ])

    val_augs = solt.Stream([
        slt.Pad((540, 540)),
        slt.Crop((512, 512), crop_mode='c'),
    ])

    return train_augs, val_augs


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


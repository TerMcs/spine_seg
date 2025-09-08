import logging
import hydra

from omegaconf import DictConfig

from data.get_masks import write_masks
from train.pipeline import train_model

log = logging.getLogger(__name__)


@hydra.main(config_path="config/", config_name="configs.yaml")
def main(cfg: DictConfig):
    if cfg.masks:  # Write updated masks to file from new CVAT xml.
        write_masks(cfg)
        log.info(f'New masks added to {cfg.mask_dir}/ from {cfg.cvat_xml}')
        print('Done.')
    else:
        log.info("No new masks written to file.")

    if cfg.train:
        log.info('Model is training.')
        train_model(cfg)
        log.info('Training complete.')


if __name__ == "__main__":
    main()

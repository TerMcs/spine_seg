import cv2
import numpy as np
from lxml import etree
from tqdm import tqdm
import pathlib

from omegaconf import DictConfig


# LABEL_MAPPING = {'BG': 0,
#                  'a_S1-VB': 1,
#                  'b_L5-S1-IVD': 2,
#                  'c_L5-VB': 3,
#                  'd_L4-L5-IVD': 4,
#                  'e_L4-VB': 5,
#                  'f_L3-L4-IVD': 6,
#                  'g_L3-VB': 7,
#                  'h_L2-L3-IVD': 8,
#                  'i_L2-VB': 9,
#                  'j_L1-L2-IVD': 10,
#                  'k_L1-VB': 11,
#                  'l_T12-L1-IVD': 12,
#                  'm_T12-VB': 13}

LABEL_MAPPING = {'1': 0,
                 '28': 1,
                 '53': 2,
                 '76': 3,
                 '96': 4,
                 '116': 5,
                 '135': 6,
                 '153': 7,
                 '171': 8,
                 '189': 9,
                 '204': 10,
                 '216': 11,
                 '224': 12,
                 '231': 13} # Mapping matches second channel of RGB machine predictions


def retrieve_polygon_from_xml(root, image_name):
    image_name_attr = ".//image[@name='{}']".format(image_name)
    image_attr = list(root.iterfind(image_name_attr))
    if len(image_attr) != 1:
        return None

    image_annotation_meta = {key: value for key, value in image_attr[0].items()}
    image_annotation_meta['shapes'] = []
    for poly_tag in image_attr[0].iter('polygon'):
        polygon = {'type': 'polygon'}
        for key, value in poly_tag.items():
            polygon[key] = value
        image_annotation_meta['shapes'].append(polygon)
    image_annotation_meta['shapes'].sort(key=lambda x: LABEL_MAPPING[x['label']])
    return image_annotation_meta


def create_mask(meta):
    mask = np.zeros((int(meta['height']), int(meta['width']), 3), dtype=np.uint8)
    for shape in meta['shapes']:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points.astype(int)

        mask = cv2.drawContours(mask, [points], -1, color=(LABEL_MAPPING[shape['label']], 0, 0))
        mask = cv2.fillPoly(mask, [points], color=(LABEL_MAPPING[shape['label']], 0, 0))

    return mask


def write_masks(cfg: DictConfig):
    cvat_xml = pathlib.Path(cfg.cvat_xml)
    image_dir = pathlib.Path(cfg.image_dir)
    mask_dir = pathlib.Path(cfg.mask_dir)

    mask_dir.mkdir(exist_ok=True)

    root = etree.parse(str(cvat_xml)).getroot()
    img_list = list(image_dir.glob('*.png'))
    for img_path in tqdm(img_list, desc='Writing contours:'):
        image_name_attr = ".//image[@name='{}']".format(img_path.name)
        image_meta_dict = retrieve_polygon_from_xml(root, img_path.name)
        if image_meta_dict is None:
            continue
        mask = create_mask(image_meta_dict)
        cv2.imwrite(str(mask_dir / img_path.name), mask)
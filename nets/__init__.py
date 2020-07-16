# __init__.py

import logging
from .sekd import SEKD, SEKDNet

__all__ = ['SEKD', 'SEKDLarge']

def get_sekd_model(
    model_name, weights_path, confidence_threshold = 0.55, nms_radius = 4,
    max_keypoints = 500, cuda = False, multi_scale = False,
    sub_pixel_location = False):
    # Create & init model.
    if model_name == 'SEKD':
        model = SEKD(weights_path, confidence_threshold, nms_radius,
                     max_keypoints, cuda, multi_scale, sub_pixel_location)
    else:
        logging.critical('Unknown model: {}'.format(model_name))

    return model


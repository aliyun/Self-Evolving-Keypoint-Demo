# __init__.py

import logging
from .sekd import SEKD
from .sekd_large import SEKDLarge
from .sekd_mobile import SEKDMobile
from .sekd_motion import SEKDMotion
from .sekd_nas import SEKDMobileCV2, SEKDMobile2CV2
from .sekd_scale import SEKDScale
from .sekd_unet import SEKDUNet

__all__ = [
    'SEKD', 'SEKDLarge', 'SEKDMobile', 'SEKDMotion',
    'SEKDMobileCV2', 'SEKDMobile2CV2', 'SEKDScale',
    'SEKDUNet']

def get_sekd_model(model_name):
    # Create & init model.
    if model_name == 'SEKD':
        model = SEKD()
    elif model_name == 'SEKDLarge':
        model = SEKDLarge()
    elif model_name == 'SEKDMotion':
        model = SEKDMotion()
    elif model_name == 'SEKDScale':
        model = SEKDScale()
    elif model_name == 'SEKDMobile':
        model = SEKDMobile()
    elif model_name == 'SEKDMobileCV2':
        model = SEKDMobileCV2()
    elif model_name == 'SEKDMobile2CV2':
        model = SEKDMobile2CV2()
    elif model_name == 'SEKDUNet':
        model = SEKDUNet()
    else:
        logging.critical('Unknown model: {}'.format(model_name))

    return model


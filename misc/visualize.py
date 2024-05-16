# -*- coding: utf-8 -*-

import numpy as np


def colorize(image, label, alpha=0.7):
    palette = ((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255),
               (255, 255, 0), (255, 0, 255), (0, 255, 255))

    if len(image.shape) == 2:  # convert to RGB three channels.
        image = np.repeat(image[:, :, np.newaxis], axis=2, repeats=3)

    n_label = np.unique(label)
    assert len(n_label) <= len(palette), 'Not enough palette to colorized!'
    mask = np.array(image)
    for nl in n_label:
        if nl == 0:
            continue
        mask[label == nl] = palette[int(nl)]

    image = alpha * image + (1 - alpha) * mask
    return image.astype(np.uint8)


def count_param_number(model, verbose=False, logger=None):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        s = '#params of %s: %.2fM' % (model.__class__.__name__, n / 1e6)
        if logger is not None: 
            logger.info(s)
        else:
            print(s)
    return n 

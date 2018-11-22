import pandas as pd
import logging as log
import glob
import numpy as np
import os.path as path
from scipy import misc

def prepare_dataset(folder):
    log.info("loading images from %s ...", folder)

    # read all images into an array
    paths    = glob.glob(path.join(folder, '*.png'))
    images   = [misc.imread(path) for path in paths]
    images   = np.asarray(images)
    # normalize pixels to [0.0, 1.0]
    images   = images / 255
    n_images = images.shape[0]
    labels   = np.zeros(n_images)

    log.info("vectorializing %d samples ...", n_images)

    # get label from filename
    for i in range(n_images):
        filename = path.basename(paths[i])[0]
        labels[i] = int(filename[0])

    # create the flattened training matrix
    dataset = []
    for i in range(n_images):
        X = images[i].flatten()
        y = labels[i]
        # label first
        dataset.append(np.insert(X, 0, y, axis=0))

    return pd.DataFrame(dataset)

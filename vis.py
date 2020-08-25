import argparse
import json
import pathlib
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from models import *
from metrics import get_metric
# from .metrics import get_metric_experimental


# Move this to Arguments later
model_save_dir = pathlib.Path('./saved-models')


def get_vis_data(input_size=(256, 256, 3), imgs=None):
    '''
    Prepare Numpy Dataset batch
    Returns a list of tuple of the PIL images and their original shapes
    '''
    if imgs is None:
        raise ValueError('Invalid Dataset directory paths provided')
    
    if not isinstance(imgs, pathlib.Path):
        imgs = pathlib.Path(imgs)

    data = {
        'dir': imgs,
        'data': []
    }

    # Take only input images
    img_names = [img_name for img_name in list(os.listdir(imgs)) if not img_name.startswith('OUT_')]
    for img_name in img_names:
        img = Image.open(imgs/img_name)
        data['data'].append((img, img.size, img_name))

    return data


def visualize(vis_data, input_size=(256, 256, 3), load_name=None):
    if load_name is not None and not os.path.exists(model_save_dir/load_name):
        raise ValueError('No saved model with the name \'%s\' exists!' % load_name)

    load_path = None
    if load_name is not None:
        load_path = model_save_dir/load_name

    model = build_inference_model(input_size=input_size, load_path=load_path)
    model.trainable = False

    # opt_adam = tf.keras.optimizers.Adam(
    #     learning_rate=0.001, beta_1=0.9, beta_2=0.999
    #     )
    # model.compile(optimizer=opt_adam, loss=recon_loss(), metrics=[
    #     get_metric('psnr'),
    #     get_metric('ssim')
    #     ])
    model.summary()

    for datum in vis_data['data']:
        img, orig_shape, img_name = datum
        # Reshape `img` to (256, 256, 3). Default method: Nearest Neighbour
        img = img.resize(input_size[:2])
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        img = model(img, training=False)

        img = np.squeeze(img, axis=0)
        img = tf.keras.preprocessing.image.array_to_img(img)
        img = img.resize(orig_shape)

        # Save image
        img.save(vis_data['dir']/('OUT_%s'%img_name))

    print('Saved output images.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Args')
    parser.add_argument('data_path', metavar='I', default='../dataset/test/imgs/', help='Path to the test directory containing input images')
    parser.add_argument('--load-name', default=None, dest='load_name', help='Name of already saved model to load')
    args = parser.parse_args()

    vis_data = get_vis_data(input_size=(256, 256,3), imgs=args.data_path)
    visualize(vis_data, input_size=(256, 256,3), load_name=args.load_name)

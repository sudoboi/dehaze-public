import argparse
import pathlib
import tensorflow as tf

from models import *


def get_train_dataset(input_size=(256, 256, 3), imgs=None, labels=None, batch_size=1, cache=False):
    '''
    Prepare DataLoader
    The dataset is catered towards (input, target)=(IMG, IMG) pairs.
    Input and Target images are put in separate folder
    '''
    if imgs is None or labels is None:
        raise ValueError('Invalid Dataset directory paths provided')
    
    if not isinstance(imgs, pathlib.Path):
        imgs = pathlib.Path(imgs)

    if not isinstance(labels, pathlib.Path):
        labels = pathlib.Path(labels)

    def _get_label_path(path):
        file_name = tf.strings.split(path, '/')[-1]
        
        # ext = tf.strings.split(file_name, '.')[-1]
        # Labels in Reside dataset are in png format

        # Reside-Dehaze dataset specific file-naming exploit
        file_id = tf.strings.split(file_name, '_')[0]

        return tf.strings.join([str(labels)+'/', file_id, '.png'])
        #return tf.strings.format('%s/{}.{}'%str(labels), (file_id, ext))

    def _get_img(path):
        # Read image path and return TF tensor
        img = tf.io.read_file(path, name='Read-Image')
        img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32, name='Decode-Image', expand_animations=False)
        return img

    def _process_path(file_path):
        label_path = _get_label_path(file_path)
        label = _get_img(label_path)
        img = _get_img(file_path)
        return img, label

    list_ds = tf.data.Dataset.list_files(str(imgs/'*'))
    ds = list_ds.map(_process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Preprocessing
    def _preprocess_images(img, label):
        '''
        * Concatenate img and label along the channels axis for consistent random cropping
        * Convert Image value range from [0,1] to [-1, 1]
        * Random Resize Image and Crop (Random Jitter)
        '''
        combined = tf.concat([img, label], axis=2)

        combined = combined*2 - 1
        
        combined = tf.image.resize(combined, (286, 286), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        combined = tf.image.random_crop(combined, (256, 256, 6))

        img = combined[:,:,:3]
        label = combined[:,:,3:]

        # Model requires label also to be passed as input
        return ((img, label), label)

    ds = ds.map(lambda img, label: _preprocess_images(img, label))

    # Batching and Optimizations
    if cache:
        ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def train(dataset, input_size=(256, 256, 3), load_path=None, save_path=None, epochs=25):
    '''
    Train the model in 2 phases.
    * Decom training Phase - Only DecomNet is trainable
    * Recon training Phase - Only DehazeNet and EnhanceNet are trainable
    '''

    if save_path is None:
        raise ValueError('Model Save Path not specified!')

    if load_path is not None and not isinstance(load_path, pathlib.Path):
        load_path = pathlib.Path(load_path)

    if not isinstance(save_path, pathlib.Path):
        save_path = pathlib.Path(save_path)

    model = build_model(input_size=input_size)

    # Load model - load only weights
    if load_path is not None:
        model = tf.keras.models.load_model(load_path, compile=False)

    ## Decom Phase
    # Make only DecomNet trainable
    model.get_layer('DecomNet').trainable = True
    model.get_layer('DehazeNet').trainable = False
    model.get_layer('EnhanceNet').trainable = False

    decom_train_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('DecomCombine').output)
    
    opt_adam = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999
        )
    decom_train_model.compile(optimizer=opt_adam, loss=decom_loss())
    decom_train_model.summary()
    
    decom_train_model.fit(dataset, epochs=epochs)

    ## Recon Phase
    # Make only DehazeNet and EnhanceNet trainable
    model.get_layer('DecomNet').trainable = False
    model.get_layer('DehazeNet').trainable = True
    model.get_layer('EnhanceNet').trainable = True

    recon_train_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('ReconFinal').output)

    opt_adam = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999
        )
    recon_train_model.compile(optimizer=opt_adam, loss=recon_loss())
    recon_train_model.summary()

    recon_train_model.fit(dataset, epochs=epochs)

    # Save model - save only weights
    model.save(save_path, save_format='h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Args')
    parser.add_argument('data_path', metavar='I', default='../dataset/imgs/', help='Path to the directory containing input images')
    parser.add_argument('label_path', metavar='L', default='../dataset/labels/', help='Path to the directory containing labels')
    parser.add_argument('--load-path', default=None, dest='load_path', help='Path to load already saved model')
    parser.add_argument('--save-path', default='./saved-models/default-model.h5', dest='save_path', help='Path to save the model')
    parser.add_argument('--batch-size', type=int, default=1, dest='batch_size', help='Number of images fed to model at once')
    parser.add_argument('--epochs', type=int, default=25, dest='epochs', help='Number of epochs')
    parser.add_argument('--cache-ds', action='store_true', dest='cache', help='Whether to cache TF Dataset')
    args = parser.parse_args()

    dataset = get_train_dataset(input_size=(256, 256,3), imgs=args.data_path, labels=args.label_path, batch_size=args.batch_size, cache=args.cache)
    train(dataset, input_size=(256, 256, 3), load_path=args.load_path, save_path=args.save_path, epochs=args.epochs)

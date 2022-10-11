import tensorflow as tf


def resize_nn(input_tensor, target_shape):
    return tf.compat.v1.image.resize_nearest_neighbor(input_tensor, target_shape[1:3])


def EnhanceNet(inputs=None, input_size=(256, 256, 4)):
    '''
    Conv Layers for enhancing Illumination (1C).
    Idea: Brighten the low illumination regions.
    '''
    if inputs is None:
        inputs = tf.keras.layers.Input(input_size)

    conv0 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=None)(inputs)
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(conv0)
    conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(conv1)
    conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(conv2)

    conv2_shape = tf.keras.backend.int_shape(conv2)
    up1 = tf.keras.layers.Lambda(resize_nn, arguments={'target_shape':conv2_shape})(conv3)
    deconv1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(up1) + conv2

    conv1_shape = tf.keras.backend.int_shape(conv1)
    up2 = tf.keras.layers.Lambda(resize_nn, arguments={'target_shape':conv1_shape})(deconv1)
    deconv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(up2) + conv1

    conv0_shape = tf.keras.backend.int_shape(conv0)
    up3 = tf.keras.layers.Lambda(resize_nn, arguments={'target_shape':conv0_shape})(deconv2)
    deconv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(up3) + conv0

    deconv3_shape = tf.keras.backend.int_shape(deconv3)
    deconv1_resized = tf.keras.layers.Lambda(resize_nn, arguments={'target_shape':deconv3_shape})(deconv1)
    deconv2_resized = tf.keras.layers.Lambda(resize_nn, arguments={'target_shape':deconv3_shape})(deconv2)

    merged = tf.keras.layers.Concatenate()([deconv1_resized, deconv2_resized, deconv3])
    merged_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=1, padding='same', activation=None)(merged)
    output = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(merged_conv)

    model = tf.keras.Model(inputs = inputs, outputs= output, name='EnhanceNet')

    return model


if __name__ == '__main__':
    model = EnhanceNet()
    tf.keras.utils.plot_model(
        model,
        to_file='../plots/%s.png' % model.name,
        show_shapes=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False
        )
    print('Plotted Model: %s' % model.name)

import tensorflow as tf

# def ssim(y_true, y_pred):
#   return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


def BottleneckBlock(inputs, out_c, droprate=0.0):
    with tf.keras.backend.name_scope('bottleneck_block'):
        inter_c = out_c * 4
        bn1 = tf.keras.layers.BatchNormalization()(inputs)
        relu1 = tf.keras.layers.ReLU()(bn1)
        conv1 = tf.keras.layers.Conv2D(filters=inter_c, kernel_size=(1,1), strides=(1,1), padding='valid', use_bias=False)(relu1)

        if droprate > 0:
            out = tf.keras.layers.SpatialDropout2D(droprate)(conv1)
        else:
            out = conv1

        bn2 = tf.keras.layers.BatchNormalization()(out)
        relu2 = tf.keras.layers.ReLU()(bn2)
        padd2 = tf.keras.layers.ZeroPadding2D(padding=(1,1))(relu2)
        conv2 = tf.keras.layers.Conv2D(filters=out_c, kernel_size=(3,3), strides=(1,1), padding='valid', use_bias=False)(padd2)

        if droprate > 0:
            out = tf.keras.layers.SpatialDropout2D(droprate)(conv2)
        else:
            out = conv2

    return out


def TransitionBlock(inputs, out_c, droprate=0.0):
    with tf.keras.backend.name_scope('transition_block'):
        bn1 = tf.keras.layers.BatchNormalization()(inputs)
        relu1 = tf.keras.layers.ReLU()(bn1)
        conv1 = tf.keras.layers.Conv2DTranspose(out_c, kernel_size=(1,1), strides=(1,1), padding='valid', use_bias=False)(relu1)
        if droprate > 0:
            out = tf.keras.layers.SpatialDropout2D(droprate)
        else:
            out = conv1
        out = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(out)

    return out


def DehazeNet(inputs=None, pretrained_weights=None, input_size=(256,256,3)):
    '''
    DCPDN for enhancing Reflectance (3C).
    Idea: Haze does not modify Illumination.
    '''
    if inputs is None:
        inputs = tf.keras.layers.Input(input_size)

    densenet121 = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs, input_shape=input_size, pooling=None)

    # ---ENCODER NETWORK---
    block3 = densenet121.get_layer('pool4_pool').output
    block2 = densenet121.get_layer('pool3_pool').output
    block1 = densenet121.get_layer('pool2_pool').output
    layer1 = densenet121.get_layer('pool1').output

    # ---DECODER NETWORK---
    dense_block4 = BottleneckBlock(block3, 256)
    trans_block4 = TransitionBlock(dense_block4, 128)
    merge4 = tf.keras.layers.Concatenate()([trans_block4, block2])

    dense_block5 = BottleneckBlock(merge4, 256)
    trans_block5 = TransitionBlock(dense_block5, 128)
    merge5 = tf.keras.layers.Concatenate()([trans_block5, block1])

    dense_block6 = BottleneckBlock(merge5, 128)
    trans_block6 = TransitionBlock(dense_block6, 64)

    dense_block7 = BottleneckBlock(trans_block6, 64)
    trans_block7 = TransitionBlock(dense_block7, 32)

    dense_block8 = BottleneckBlock(trans_block7, 32)
    trans_block8 = TransitionBlock(dense_block8, 16)
    merge8 = tf.keras.layers.Concatenate()([trans_block8, densenet121.input])

    conv9 = tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(merge8)
    conv9 = tf.keras.layers.LeakyReLU(0.2)(conv9)

    # ---PYRAMID POOLING NETWORK---
    x101 = tf.keras.layers.AveragePooling2D(pool_size=(32,32))(conv9) #output will be 1/32 times conv9
    x102 = tf.keras.layers.AveragePooling2D(pool_size=(16,16))(conv9) #output will be 1/16 times conv9
    x103 = tf.keras.layers.AveragePooling2D(pool_size=(8,8))(conv9) #output will be 1/8 times conv9
    x104 = tf.keras.layers.AveragePooling2D(pool_size=(4,4))(conv9) #output will be 1/4 times conv9

    conv1010 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='valid', use_bias=True)
    conv1020 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='valid', use_bias=True)
    conv1030 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='valid', use_bias=True)
    conv1040 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='valid', use_bias=True)

    x1010 = tf.keras.layers.LeakyReLU(0.2)(conv1010(x101))
    x1020 = tf.keras.layers.LeakyReLU(0.2)(conv1020(x102))
    x1030 = tf.keras.layers.LeakyReLU(0.2)(conv1030(x103))
    x1040 = tf.keras.layers.LeakyReLU(0.2)(conv1040(x104))

    def UpSampling_custom(x):
        x = tf.image.resize(x,size=(input_size[0], input_size[1]), method='nearest')
        return x

    x1010_layer = tf.keras.layers.Lambda(UpSampling_custom)
    x1020_layer = tf.keras.layers.Lambda(UpSampling_custom)
    x1030_layer = tf.keras.layers.Lambda(UpSampling_custom)
    x1040_layer = tf.keras.layers.Lambda(UpSampling_custom)

    x1010 = x1010_layer(x1010)
    x1020 = x1020_layer(x1020)
    x1030 = x1030_layer(x1030)
    x1040 = x1040_layer(x1040)

    dehaze = tf.keras.layers.Concatenate()([x1010, x1020, x1030, x1040, conv9])
    dehaze = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(dehaze)
    dehaze = tf.keras.layers.Activation('sigmoid')(dehaze)

    out = dehaze
    
    model = tf.keras.Model(inputs=inputs, outputs=out, name='DehazeNet')
    
    return model


if __name__ == '__main__':
    model = DehazeNet()
    tf.keras.utils.plot_model(
        model,
        to_file='../plots/%s.png' % model.name,
        show_shapes=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False
        )
    print('Plotted Model: %s' % model.name)

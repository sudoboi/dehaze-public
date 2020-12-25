import tensorflow as tf


def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m


def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu', 
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='tanh')(o)
    return Model(inputs=i, outputs=o)

def DecomNet(inputs=None, input_size=(256, 256, 3)):
    '''
    UNet for the normal 3 channel images and output is Reflectance (3C) and Illumination (1C).
    '''
    unet = UNet(
        input_size,
        out_ch=4,
        start_ch=64,
        batchnorm=True,
        residual=True
        )
    outputs = unet(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='DecomNet')


if __name__ == '__main__':
    model = DecomNet()
    tf.keras.utils.plot_model(
        model,
        to_file='../plots/%s.png' % model.name,
        show_shapes=False,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False
        )
    print('Plotted Model: %s' % model.name)

import tensorflow as tf

from .decom import DecomNet
from .dehaze import DehazeNet
from .enhance import EnhanceNet


def decom_loss():
    '''
    Decom Loss function adapted from DeepRetinex
    '''
    def concat(layers):
        return tf.concat(layers, axis=-1)

    def gradient(input_tensor, direction):
        smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(input_tensor, direction):
        return tf.nn.avg_pool2d(gradient(input_tensor, direction), ksize=3, strides=1, padding='SAME')

    def smooth(input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(gradient(input_I, "x") * tf.exp(-10 * ave_gradient(input_R, "x")) + gradient(input_I, "y") * tf.exp(-10 * ave_gradient(input_R, "y")))

    def loss_fn(_y_true, _y_pred):
        y_true = _y_pred[:,:,:,:4]
        y_pred = _y_pred[:,:,:,4:]

        I_low = y_pred[:,:,:,3:4]
        I_high = y_true[:,:,:,3:4]
        R_low = y_pred[:,:,:,0:3]
        R_high = y_true[:,:,:,0:3]

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])

        output_R_low = R_low
        output_I_low = I_low_3

        # loss
        recon_loss_low = tf.math.reduce_mean(tf.math.abs(R_low * I_low_3))
        recon_loss_high = tf.math.reduce_mean(tf.math.abs(R_high * I_high_3))
        recon_loss_mutal_low = tf.math.reduce_mean(tf.math.abs(R_high * I_low_3))
        recon_loss_mutal_high = tf.math.reduce_mean(tf.math.abs(R_low * I_high_3 ))
        equal_R_loss = tf.math.reduce_mean(tf.math.abs(R_low - R_high))

        Ismooth_loss_low = smooth(I_low, R_low)
        Ismooth_loss_high = smooth(I_high, R_high)

        loss_Decom = recon_loss_low + recon_loss_high + \
            0.001 * recon_loss_mutal_low + 0.001 * recon_loss_mutal_high + \
            0.1 * Ismooth_loss_low + 0.1 * Ismooth_loss_high + \
            0.01 * equal_R_loss

        return loss_Decom

    return loss_fn


def recon_loss():
    '''
    Recon Loss function adapted from Perceptual Loss
    '''
    def perceptual_loss(y_true, _y_pred,
        model_type='vgg', input_shape=(256, 256, 3), layers=None, weights=list((8, 4, 2, 1))):
        
        y_pred = _y_pred[0]

        # set up base model
        if model_type == 'vgg':
            from tensorflow.keras.applications.vgg16 import VGG16   
            base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            if layers is None:
                layers = [2, 5, 9] 
        elif model_type == 'xception':
            from tensorflow.keras.applications.xception import Xception
            base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
            if layers is None:
                layers = [19, 42, 62]
        else:
            raise NotImplementedError()
        
        # set up loss model
        outputs = [base_model.layers[idx].output for idx in layers]
        loss_model = Model(inputs=base_model.input, outputs=outputs)
        loss_model.trainable = False

        # extract y true and predicted features
        y_true_features = loss_model(y_true)
        y_pred_features = loss_model(y_pred)

        # calculate weighted loss
        loss = weights[0] * K.mean(K.square(y_true - y_pred))
        for idx in range(0, len(weights) - 1):
            loss += weights[idx + 1] * K.mean(K.square(y_true_features[idx] - y_pred_features[idx]))
        loss = loss / sum(weights)
        
        return loss

    return perceptual_loss

def build_model(input_size=(256, 256, 3)):
    # x = Hazed, Low-light input image
    # y = Dehazed, Illuminated ground truth image
    x = tf.keras.layers.Input(input_size)
    y = tf.keras.layers.Input(input_size)

    # Decomposition
    # z_R = z[:,:,:,:3]
    # z_I = z[:,:,:,3:4]
    decomNet = DecomNet(input_size=input_size)
    x_decom = decomNet(x)
    y_decom = decomNet(y)
    decomCombine = tf.keras.layers.Lambda(lambda z: tf.concat([z[0], z[1]], axis=-1), name='DecomCombine')((y_decom, x_decom))

    dehazeNet = DehazeNet(input_size=input_size)
    x_R_dehazed = dehazeNet(x_decom[:,:,:,:3])

    enhanceNet = EnhanceNet(input_size=input_size[:-1]+(4,))
    x_I_illum = enhanceNet(x_decom)

    def recon_mul(dcpdn_out, enh_net_out):
        enh_net_out_3 = tf.concat([enh_net_out, enh_net_out, enh_net_out], axis=-1)
        recon = dcpdn_out * enh_net_out_3
        return recon

    y_hat = tf.keras.layers.Lambda(lambda x: recon_mul(x[0], x[1]), name = 'ReconFinal') ((x_R_dehazed, x_I_illum))

    combined_model = tf.keras.Model(inputs=[x, y], outputs=[y_hat, decomCombine])

    return combined_model

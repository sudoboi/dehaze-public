import tensorflow as tf

try:
    import tensorflow_addons.metrics as tf_metrics
except:
    # Might be Unstable
    import tensorflow.python.keras.metrics as tf_metrics


class MeanMetricWrapper(tf.keras.metrics.Metric):

    def __init__(self, func, name, **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='%s_total' % name, initializer='zeros')
        self.count = self.add_weight(name='%s_count' % name, initializer='zeros')
        self.func = func

    def update_state(self, y_true, y_pred, sample_weight=None):
        assert y_true.shape.as_list() == y_pred.shape.as_list(), '`y_true` and `y_pred` shapes mismatch'

        values = self.func(y_true, y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, y_true)
            values = tf.multiply(values, sample_weight)

        self.total.assign_add(tf.math.reduce_sum(values))

        if sample_weight is not None:
            num_vals = tf.math.reduce_sum(sample_weight)
        else:    
            num_vals = tf.size(values)
        self.count.assign_add(num_vals)

    def result(self):
        return tf.math.div_no_nan(self.total, self.count)


def _psnr_metric_core(y_true, y_pred):
    '''
    y_true and y_pred have a range [-1, 1]
    Transform them to [0, 1]
    '''
    y_true = (y_true + 1.) / 2.
    y_pred = (y_pred + 1.) / 2.
    return tf.image.psnr(y_true, y_pred, max_val=1.)


def _ssim_metric_core(y_true, y_pred):
    '''
    y_true and y_pred have a range [-1, 1]
    Transform them to [0, 1]
    '''
    y_true = (y_true + 1.) / 2.
    y_pred = (y_pred + 1.) / 2.
    return tf.image.ssim(y_true, y_pred, max_val=1.,
        filter_size=11, filter_sigma=1.5,
        k1=0.01, k2=0.03
        )


def get_metric_experimental(metric):
    if metric == 'psnr':
        return tf_metrics.MeanMetricWrapper(_psnr_metric_core, 'psnr_metric', dtype=tf.float32)
    elif metric == 'ssim':
        return tf_metrics.MeanMetricWrapper(_ssim_metric_core, 'ssim_metric', dtype=tf.float32)
    else:
        raise NotImplementedError('Metric \'%s\' is not implemented!' % metric)


def get_metric(metric):
    if metric == 'psnr':
        return MeanMetricWrapper(_psnr_metric_core, 'psnr_metric', dtype=tf.float32)
    elif metric == 'ssim':
        return MeanMetricWrapper(_ssim_metric_core, 'ssim_metric', dtype=tf.float32)
    else:
        raise NotImplementedError('Metric \'%s\' is not implemented!' % metric)


# class MetricCallback(tf.keras.callbacks.Callback):

#     def on_train_begin(self, logs={}):
#         pass

#     def on_epoch_end(self, batch, logs={}):
#         pass

#     def get_data(self):
#         pass

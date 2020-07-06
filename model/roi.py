import tensorflow as tf


def roi_pooling(feature, rois, img_size, pool_size):
    """
    用tf.image.crop_and_resize实现roi_align
    :param feature: 特征图[1, hh, ww, c]
    :param rois: 原图的rois
    :param img_size: 原图的尺寸
    :param pool_size: align后的尺寸
    """

    # 所有需要pool的框在batch中的对应图片序号，由于batch_size为1，因此box_ind里面的值都为0
    box_ind = tf.zeros(rois.shape[0], dtype=tf.int32)
    # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]

    # 在这里取到归一化框的坐标时需要的图片尺度
    normalization = tf.cast(tf.stack([img_size[0], img_size[1], img_size[0], img_size[1]], axis=0), dtype=tf.float32)
    # 归一化框的坐标为原图的0~1倍尺度
    boxes = rois / normalization

    # 进行ROI pool，之所以需要归一化框的坐标是因为tf接口的要求
    # 2000,7,7,256
    pool = tf.image.crop_and_resize(feature, boxes, box_ind, crop_size=pool_size)
    return pool


class RoIPooling2D(tf.keras.Model):

    def __init__(self, pool_size):
        super(RoIPooling2D, self).__init__()
        self.pool_size = pool_size

    def __call__(self, feature, rois, img_size):
        return roi_pooling(feature, rois, img_size, self.pool_size)


class RoIHead(tf.keras.Model):

    def __init__(self, n_class, pool_size):
        # n_class includes the background
        super(RoIHead, self).__init__()

        self.fc = tf.keras.layers.Dense(4096)
        self.cls_loc = tf.keras.layers.Dense(n_class * 4)
        self.score = tf.keras.layers.Dense(n_class)

        self.n_class = n_class
        self.roi = RoIPooling2D(pool_size)

    def __call__(self, feature, rois, img_size, training=None):

        rois = tf.constant(rois, dtype=tf.float32)
        pool = self.roi(feature, rois, img_size)
        pool = tf.reshape(pool, [rois.shape[0], -1])
        fc = self.fc(pool)
        roi_cls_locs = self.cls_loc(fc)
        roi_scores = self.score(fc)

        return roi_cls_locs, roi_scores

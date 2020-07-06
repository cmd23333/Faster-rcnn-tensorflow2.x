import tensorflow as tf
import numpy as np
from model.rpn import RegionProposalNetwork, Extractor
from model.roi import RoIHead
from utils.anchor import loc2bbox, AnchorTargetCreator, ProposalTargetCreator


def _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma):
    # pred_loc, gt_loc, in_weight
    sigma2 = sigma ** 2
    sigma2 = tf.constant(sigma2, dtype=tf.float32)
    diff = in_weight * (pred_loc - gt_loc)
    abs_diff = tf.math.abs(diff)
    abs_diff = tf.cast(abs_diff, dtype=tf.float32)
    flag = tf.cast(abs_diff.numpy() < (1./sigma2), dtype=tf.float32)
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return tf.reduce_sum(y)


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    """
    :param pred_loc: 1,38,50,36
    :param gt_loc: 17100,4
    :param gt_label: 17100
    """
    idx = gt_label > 0
    idx = tf.stack([idx, idx, idx, idx], axis=1)
    idx = tf.reshape(idx, [-1, 4])
    in_weight = tf.cast(idx, dtype=tf.int32)
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.numpy(), sigma)
    # Normalize by total number of negative and positive rois.
    loc_loss /= (tf.reduce_sum(tf.cast(gt_label >= 0, dtype=tf.float32)))  # ignore gt_label==-1 for rpn_loss
    return loc_loss


class FasterRCNN(tf.keras.Model):

    def __init__(self, n_class, pool_size):
        super(FasterRCNN, self).__init__()
        self.n_class = n_class
        self.extractor = Extractor()
        self.rpn = RegionProposalNetwork()
        self.head = RoIHead(n_class, pool_size)
        self.score_thresh = 0.7
        self.nms_thresh = 0.3

    def __call__(self, x):
        img_size = x.shape[1:3]
        feature_map, rpn_locs, rpn_scores, rois, roi_score, anchor = self.rpn(x)
        roi_cls_locs, roi_scores = self.head(feature_map, rois, img_size)

        return roi_cls_locs, roi_scores, rois

    def predict(self, imgs):
        bboxes = []
        labels = []
        scores = []
        img_size = imgs.shape[1:3]
        # (2000,84) (2000,21) (2000,4)
        roi_cls_loc, roi_score, rois = self(imgs)
        prob = tf.nn.softmax(roi_score, axis=-1)
        prob = prob.numpy()
        roi_cls_loc = roi_cls_loc.numpy()
        roi_cls_loc = roi_cls_loc.reshape(-1, self.n_class, 4)  # 2000, 21, 4

        for label_index in range(1, self.n_class):

            cls_bbox = loc2bbox(rois, roi_cls_loc[:, label_index, :])
            # clip bounding box
            cls_bbox[:, 0::2] = tf.clip_by_value(cls_bbox[:, 0::2], clip_value_min=0, clip_value_max=img_size[0])
            cls_bbox[:, 1::2] = tf.clip_by_value(cls_bbox[:, 1::2], clip_value_min=0, clip_value_max=img_size[1])
            cls_prob = prob[:, label_index]

            mask = cls_prob > 0.05
            cls_bbox = cls_bbox[mask]
            cls_prob = cls_prob[mask]
            keep = tf.image.non_max_suppression(cls_bbox, cls_prob, max_output_size=-1, iou_threshold=self.nms_thresh)

            if len(keep) > 0:
                bboxes.append(cls_bbox[keep.numpy()])
                # The labels are in [0, self.n_class - 2].
                labels.append((label_index - 1) * np.ones((len(keep),)))
                scores.append(cls_prob[keep.numpy()])
        if len(bboxes) > 0:
            bboxes = np.concatenate(bboxes, axis=0).astype(np.float32)
            labels = np.concatenate(labels, axis=0).astype(np.float32)
            scores = np.concatenate(scores, axis=0).astype(np.float32)

        return bboxes, labels, scores


class FasterRCNNTrainer(tf.keras.Model):

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3.0
        self.roi_sigma = 1.0
        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

    def __call__(self, imgs, bbox, label, scale, training=None):
        _, H, W, _ = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs, training=training)
        rpn_locs, rpn_scores, roi, anchor = self.faster_rcnn.rpn(features, img_size, scale, training=training)

        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]

        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox.numpy(), label.numpy())
        roi_cls_loc, roi_score = self.faster_rcnn.head(features, sample_roi, img_size, training=training)

        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox.numpy(), anchor, img_size)
        gt_rpn_label = tf.constant(gt_rpn_label, dtype=tf.int32)
        gt_rpn_loc = tf.constant(gt_rpn_loc, dtype=tf.float32)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
        idx_ = gt_rpn_label != -1
        rpn_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(gt_rpn_label[idx_], rpn_score[idx_])

        # ROI losses
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = tf.reshape(roi_cls_loc, [n_sample, -1, 4])
        idx_ = [[i, j] for i, j in zip(tf.range(n_sample), tf.constant(gt_roi_label))]
        roi_loc = tf.gather_nd(roi_cls_loc, idx_)
        gt_roi_label = tf.constant(gt_roi_label)
        gt_roi_loc = tf.constant(gt_roi_loc)
        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
        idx_ = gt_roi_label != 0
        roi_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(gt_roi_label[idx_], roi_score[idx_])

        return rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss

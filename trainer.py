import datetime

from utils.config import Config
from model.fasterrcnn import FasterRCNNTrainer, FasterRCNN
import tensorflow as tf
from utils.data import Dataset

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = Config()
config._parse({})

print("读取数据中....")
dataset = Dataset(config)


frcnn = FasterRCNN(21, (7, 7))
print('model construct completed')


"""
feature_map, rpn_locs, rpn_scores, rois, roi_indices, anchor = frcnn.rpn(x, scale)
'''
feature_map : (1, 38, 50, 256) max= 0.0578503
rpn_locs    : (1, 38, 50, 36) max= 0.058497224
rpn_scores  : (1, 17100, 2) max= 0.047915094
rois        : (2000, 4) max= 791.0
roi_indices :(2000,) max= 0
anchor      : (17100, 4) max= 1154.0387
'''
bbox = bboxes
label = labels
rpn_score = rpn_scores
rpn_loc = rpn_locs
roi = rois

proposal_target_creator = ProposalTargetCreator()
sample_roi, gt_roi_loc, gt_roi_label, keep_index = proposal_target_creator(roi, bbox, label)

roi_cls_loc, roi_score = frcnn.head(feature_map, sample_roi, img_size)
'''
roi_cls_loc : (128, 84) max= 0.062198948
roi_score   : (128, 21) max= 0.045144305
'''

anchor_target_creator = AnchorTargetCreator()
gt_rpn_loc, gt_rpn_label = anchor_target_creator(bbox, anchor, img_size)
rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, 3)
idx_ = gt_rpn_label != -1
rpn_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()(gt_rpn_label[idx_], rpn_score[0][idx_])

# ROI losses
n_sample = roi_cls_loc.shape[0]
roi_cls_loc = tf.reshape(roi_cls_loc, [n_sample, -1, 4])
idx_ = [[i,j] for i,j in zip(range(n_sample), gt_roi_label)]
roi_loc = tf.gather_nd(roi_cls_loc, idx_)

roi_loc_loss = _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, 1)
roi_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()(gt_roi_label, roi_score)
"""

model = FasterRCNNTrainer(frcnn)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

epochs = 12
loss = []
for epoch in range(epochs):
    for i in range(len(dataset)):
        img, bboxes, labels, scale = dataset[1]
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)
        with tf.GradientTape() as tape:
            rpn_loc_l, rpn_cls_l, roi_loc_l, roi_cls_l = model(img, bboxes, labels, scale, training=True)
            total_loss = rpn_loc_l + rpn_cls_l + roi_loc_l + roi_cls_l
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if i % 1 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('rpn_loc_loss', float(rpn_loc_l), step=i + epoch * len(dataset))
                tf.summary.scalar('rpn_cls_loss', float(rpn_cls_l), step=i + epoch * len(dataset))
                tf.summary.scalar('roi_loc_loss', float(roi_loc_l), step=i + epoch * len(dataset))
                tf.summary.scalar('roi_cls_loss', float(roi_cls_l), step=i + epoch * len(dataset))

        if i % 1000 == 0:
            model.save_weights('frcnn.h5')


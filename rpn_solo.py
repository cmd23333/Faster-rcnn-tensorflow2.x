from utils.config import Config
from model.rpn import RegionProposalNetwork
import tensorflow as tf
from utils.data import Dataset
from utils.anchor import AnchorTargetCreator
from model.fasterrcnn import _fast_rcnn_loc_loss
from utils.visualize import vis_train

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = Config()
config._parse({})
dataset = Dataset(config)

rpn = RegionProposalNetwork()
anchor_target_creator = AnchorTargetCreator()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

img, bbox, label = dataset[1]
x = tf.convert_to_tensor(img, dtype=tf.float32)
x = tf.expand_dims(x, axis=0)
feature_map, rpn_loc, rpn_score, roi, roi_score, anchor = rpn(x)
rpn.load_weights('rpn.h5')

test_img, test_bbox, test_label = dataset[1]
test_x = tf.convert_to_tensor(test_img, dtype=tf.float32)
test_x = tf.expand_dims(test_x, axis=0)

for epoch in range(1500):
    for i in range(1):
        img, bbox, label = dataset[i]
        x = tf.convert_to_tensor(img, dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)
        with tf.GradientTape() as tape:
            feature_map, rpn_loc, rpn_score, roi, roi_score, anchor = rpn(x)

            gt_rpn_loc, gt_rpn_label = anchor_target_creator(bbox, anchor)
            rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label)

            idx_ = gt_rpn_label != -1
            rpn_cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()(gt_rpn_label[idx_], rpn_score[0][idx_])

            total_loss = rpn_cls_loss + rpn_loc_loss
        grads = tape.gradient(total_loss, rpn.trainable_variables)
        optimizer.apply_gradients(zip(grads, rpn.trainable_variables))

        if i % 20 == 0:
            print("step", i)
            print("rpn_loc_loss = ", round(float(rpn_loc_loss), 4),
                  "rpn_cls_loss = ", round(float(rpn_cls_loss), 4))

        if i % 1000 == 0:
            rpn.save_weights('rpn.h5')

    if epoch % 50 == 0:
        feature_map, rpn_loc, rpn_score, roi, roi_score, anchor = rpn(test_x)
        vis_train(test_img, test_bbox, test_label, roi, roi_score, epoch)

from utils.config import Config
from model.fasterrcnn import FasterRCNNTrainer, FasterRCNN
import tensorflow as tf
from utils.data import Dataset

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = Config()
config._parse({})
dataset = Dataset(config)
frcnn = FasterRCNN(21, (7, 7))

model = FasterRCNNTrainer(frcnn)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
# 相当于build一下模型
img, bboxes, labels, scale = dataset[1]
x = tf.convert_to_tensor(img, dtype=tf.float32)
x = tf.expand_dims(x, axis=0)
_, _, _, _ = model(x, bboxes, labels, scale)
# 然后就能载入权重了
model.load_weights('frcnn.h5')

img, bboxes, labels, scale = dataset[233]
img_size = img.shape[:2]
x = tf.convert_to_tensor(img, dtype=tf.float32)
x = tf.expand_dims(x, axis=0)

print(model.faster_rcnn.predict(x, img_size))
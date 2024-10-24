import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub

#set tensorflow threading options for cpu optimization
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

#resize the image to the expected size (256x256) and prepare for input
img_resized = tf.image.resize(img, [256, 256], method='bicubic', preserve_aspect_ratio=False)

#change from hwc to chw format
img_resized = tf.transpose(img_resized, [2, 0, 1])  

#convert the tensor to a numpy array
img_input = img_resized.numpy()  

#add batch dimension
img_input = img_input.reshape(1, 3, 256, 256)  
tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)

module = hub.load("https://www.kaggle.com/models/intel/midas/TensorFlow1/v2-1-small/1", tags=['serve'])

#perform inference using the model
output = module.signatures['serving_default'](tensor)

#convert the output to a numpy array
prediction = output['default'].numpy()
prediction = prediction.reshape(256, 256)

#resize the prediction back to the original image size
prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

#normalize the depth map for visualization
depth_min = prediction.min()
depth_max = prediction.max()
img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")

cv2.imwrite("output.png", img_out)
plt.imshow(img_out, cmap='inferno')
plt.show()
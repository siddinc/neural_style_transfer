import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from constants import (
  style_weights,
  style_weight,
  content_weight,
  style_layers,
  content_layers,
  num_content_layers,
  num_style_layers
)


def load_image(image_path):
  image = plt.imread(image_path)
  img = tf.image.convert_image_dtype(image, tf.float32)
  img = tf.image.resize(img, [400, 400])
  img = tf.expand_dims(img, axis=0)
  return img


def show_image(image):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)
  plt.imshow(image)
  plt.show()


def clip_image(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def generate_noise_image(image):
  pass


def tensor_to_image(input_tensor):
  input_tensor = input_tensor * 255
  np_img = np.array(input_tensor, dtype=np.uint8)
  if np.ndim(np_img) > 3:
    np_img = np_img[0]
  show_image(np_img)


def gram_matrix(input_tensor):
  temp = input_tensor
  temp = tf.squeeze(temp)
  temp = tf.reshape(temp, [input_tensor.shape[3], input_tensor.shape[1] * input_tensor.shape[2]])
  result = tf.linalg.matmul(temp, temp, transpose_b=True)
  result /= (input_tensor.shape[1] * input_tensor.shape[2])
  gram_matrix = tf.expand_dims(result, axis=0)
  return gram_matrix


def total_loss(outputs, style_targets, content_targets):
  style_outputs = outputs['style']
  content_outputs = outputs['content']
  # style_targets = outputs['style']
  # content_targets = outputs['content']

  style_loss = tf.add_n([style_weights[name] * tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
  style_loss *= style_weight / num_style_layers

  content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
  content_loss *= content_weight / num_content_layers

  total_loss = content_loss + style_loss
  return total_loss

if __name__ == "__main__":
  img = load_image('../images/dog.png')
  # show_image(img)
  # gram_matrix(img)
  # tensor_to_img(img)
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from constants import (
  style_layers,
  content_layers,
  steps_per_epoch,
  epochs,
  sample_image_path
)
from model_fn import StyleContentModel, train_step
from utility_fn import (
  load_image,
  tensor_to_image,
  save_image,
  show_image
)


def fit(image, extractor, style_targets, content_targets):
  epoch = 0
  start = time.time()

  for n in range(epochs):
    print("Epoch: {}".format(epoch + 1))
    for m in range(steps_per_epoch):
      print("Step: {}".format(m + 1))
      train_step(image, extractor, style_targets, content_targets)
    epoch += 1

  end = time.time()
  print("Total time: {:.2f}s".format(end - start))

  np_img = tensor_to_image(image)
  show_image(np_img)
  save_image(np_img)


if __name__ == "__main__":
  content_image = load_image(sample_image_path + '/golden_gate.jpg')
  style_image = load_image(sample_image_path + '/starry_night.jpg')

  extractor = StyleContentModel(style_layers, content_layers)
  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  image = tf.Variable(content_image)
  fit(image, extractor, style_targets, content_targets)
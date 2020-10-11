import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from constants import (
  style_layers,
  content_layers,
  steps_per_epoch,
  sample_image_path
)
from model_fn import StyleContentModel, train_step
from utility_fn import (
  load_image,
  tensor_to_image,
  save_image,
  show_image
)


def fit(image, extractor, style_targets, content_targets, no_of_epochs):
  epoch = 0
  start = time.time()

  for n in range(no_of_epochs):
    print("Epoch: {}".format(epoch + 1))
    step = ""
    for m in range(steps_per_epoch):
      print("Step: {}".format(m + 1))
      train_step(image, extractor, style_targets, content_targets)
    epoch += 1

  end = time.time()
  print("Total time: {:.2f}s".format(end - start))

  np_img = tensor_to_image(image)
  show_image(np_img)
  save_image(np_img)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-cip", "--content_img_path", required=True, help="path to content image")
  ap.add_argument("-sip", "--style_img_path", required=True, help="path to style image")
  ap.add_argument("-noe", "--no_of_epochs", required=True, help="no. of epochs")
  args = vars(ap.parse_args())

  content_image = load_image(args["content_img_path"])
  style_image = load_image(args["style_img_path"])
  no_of_epochs = int(args["no_of_epochs"])

  extractor = StyleContentModel(style_layers, content_layers)
  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  image = tf.Variable(content_image)
  fit(image, extractor, style_targets, content_targets, no_of_epochs)
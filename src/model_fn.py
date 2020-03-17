import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import PIL
import matplotlib.pyplot as plt
from utility_fn import gram_matrix, total_loss, clip_image, load_image, tensor_to_image
from constants import style_layers, content_layers

def mini_vgg(layer_names):
  vgg = VGG19(weights='imagenet', include_top=False)
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = Model([vgg.input], outputs)
  return model
  

class StyleContentModel(Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  mini_vgg(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    inputs *= 255.0
    preprocessed_input = preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    style_dict = {style_layer: style_output for style_layer, style_output in zip(self.style_layers, style_outputs)}

    content_dict = {content_layer: content_output for content_layer, content_output in zip(self.content_layers, content_outputs)}
    
    return {'content': content_dict, 'style': style_dict}
  
opt = Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image, extractor, style_targets, content_targets):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = total_loss(outputs, style_targets, content_targets)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_image(image))

if __name__ == "__main__":
  content_image = load_image('../images/dog.png')
  style_image = load_image('../images/painting.png')

  extractor = StyleContentModel(style_layers, content_layers)
  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  print(content_targets)

  # image = tf.Variable(content_image)
  # train_step(image, extractor, style_targets, content_targets)
  # train_step(image, extractor, style_targets, content_targets)
  # train_step(image, extractor, style_targets, content_targets)
  # tensor_to_image(image)
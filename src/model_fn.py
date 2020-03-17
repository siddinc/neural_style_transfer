from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from constants import (
  style_layers,
  content_layers,
  learning_rate,
  beta_1,
  epsilon
)
from utility_fn import (
  gram_matrix,
  total_loss,
  clip_image
)


opt = Adam(learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon)


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
    inputs = inputs * 255.0
    preprocessed_input = preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    style_dict = {style_layer: style_output for style_layer, style_output in zip(self.style_layers, style_outputs)}
    content_dict = {content_layer: content_output for content_layer, content_output in zip(self.content_layers, content_outputs)}
    return {'content': content_dict, 'style': style_dict}


@tf.function()
def train_step(image, extractor, style_targets, content_targets):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = total_loss(outputs, style_targets, content_targets)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_image(image))
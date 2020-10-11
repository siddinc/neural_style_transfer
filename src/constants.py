import os


sample_image_path = os.path.abspath('../images/samples')
output_image_path = os.path.abspath('../images/output')

content_layers = ['block5_conv2']
num_content_layers = len(content_layers)

style_layers = [
  'block1_conv1',
  'block2_conv1',
  'block3_conv1',
  'block4_conv1',
  'block5_conv1'
]
num_style_layers = len(style_layers)

style_weights = {
  'block1_conv1': 1.0,
  'block2_conv1': 1.0,
  'block3_conv1': 1.0,
  'block4_conv1': 1.0,
  'block4_conv2': 1.0,
  'block5_conv1': 1.0,
}

content_weight = 1e4
style_weight = 1e-2

clip_min = 0.0
clip_max = 1.0

learning_rate = 0.02
beta_1 = 0.99
epsilon = 1e-1

steps_per_epoch = 10
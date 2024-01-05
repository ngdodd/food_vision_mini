
import torch.nn as nn

def calc_shape(h, w, pad, kernel, stride):
  h_out = (h + 2*pad[0] - kernel[0])/stride[0] + 1
  w_out = (w + 2*pad[1] - kernel[1])/stride[1] + 1
  return h_out, w_out

def calc_linear_dim(h, w, conv_blocks, n_out_filters):
  conv_kernel, conv_stride, conv_padding = [3,3], [1,1], [1,1]
  pool_kernel, pool_stride, pool_padding = [2,2], [2,2], [0,0]

  for block_itr in range(conv_blocks):
    h, w = calc_shape(h, w, conv_padding, conv_kernel, conv_stride)
    h, w = calc_shape(h, w, conv_padding, conv_kernel, conv_stride)
    h, w = calc_shape(h, w, pool_padding, pool_kernel, pool_stride)

  return n_out_filters*h*w
  
class TinyVGG(nn.Module):
  def __init__(self, n_classes=3, image_height=64, image_width=64):
    super().__init__()
    input_channels, hidden_dim = 3, 10
    self.conv_block1 = self.conv_block(input_channels)
    self.conv_block2 = self.conv_block(hidden_dim)
    self.flatten = nn.Flatten()
    self.calc_linear_shape(image_height, image_width)

    linear_dim = calc_linear_dim(image_height, image_width,
                                 conv_blocks=2, n_out_filters=hidden_dim)
    self.fc = nn.Linear(in_features=int(linear_dim), out_features=n_classes)

  def forward(self, x):
    return self.fc(self.flatten(self.conv_block2(self.conv_block1(x.float()))))    

  def conv_block(self, start_channels, hidden_channels=10, out_channels=10,
                 kernel_size=3, conv_stride=1, conv_padding=1,
                 pool_kernel_size=2, pool_stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=start_channels,
                  out_channels=hidden_channels,
                  kernel_size=kernel_size,
                  stride=conv_stride,
                  padding=conv_padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=conv_stride,
                  padding=conv_padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=pool_kernel_size,
                     stride=pool_stride)
    )

class TinyVGGWithBatchNorm(nn.Module):
  def __init__(self, n_classes=3, image_height=64, image_width=64):
    super().__init__()
    input_channels, hidden_dim = 3, 10
    self.conv_block1 = self.conv_block(input_channels)
    self.conv_block2 = self.conv_block(hidden_dim)
    self.flatten = nn.Flatten()

    linear_dim = calc_linear_dim(image_height, image_width,
                                 conv_blocks=2, n_out_filters=hidden_dim)
    self.fc = nn.Linear(in_features=int(linear_dim), out_features=n_classes)

  def forward(self, x):
    return self.fc(self.flatten(self.conv_block2(self.conv_block1(x.float()))))

  def conv_block(self, start_channels, hidden_channels=10, out_channels=10,
                 kernel_size=3, conv_stride=1, conv_padding=1,
                 pool_kernel_size=2, pool_stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=start_channels,
                  out_channels=hidden_channels,
                  kernel_size=kernel_size,
                  stride=conv_stride,
                  padding=conv_padding),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=conv_stride,
                  padding=conv_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=pool_kernel_size,
                     stride=pool_stride)
    )

class MediumVGG(nn.Module):
  def __init__(self, n_classes=3, image_height=64, image_width=64):
    super().__init__()
    self.conv_blocks = nn.Sequential(
        self.conv_block(3, 10, 10),
        self.conv_block(10, 20, 20),
        self.conv_block(20, 40, 40),
        self.conv_block(40, 80, 80),
        self.conv_block(80, 160, 160)
    )
    self.flatten = nn.Flatten()

    linear_dim = calc_linear_dim(image_height, image_width,
                                 conv_blocks=5, n_out_filters=160)
    self.fc = nn.Linear(in_features=int(linear_dim), out_features=n_classes)

  def forward(self, x):
    return self.fc(self.flatten(self.conv_blocks(x.float())))

  def conv_block(self, start_channels, hidden_channels=10, out_channels=10,
                 kernel_size=3, conv_stride=1, conv_padding=1,
                 pool_kernel_size=2, pool_stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=start_channels,
                  out_channels=hidden_channels,
                  kernel_size=kernel_size,
                  stride=conv_stride,
                  padding=conv_padding),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=conv_stride,
                  padding=conv_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=pool_kernel_size,
                     stride=pool_stride)
    )

class MediumVGGNormalized(nn.Module):
  def __init__(self, n_classes=3, image_height=64, image_width=64):
    super().__init__()
    self.conv_blocks = nn.Sequential(
        self.conv_block(3, 10, 10),
        self.conv_block(10, 20, 20),
        self.conv_block(20, 40, 40),
        self.conv_block(40, 80, 80),
        self.conv_block(80, 160, 160)
    )
    self.flatten = nn.Flatten()

    linear_dim = calc_linear_dim(image_height, image_width,
                                 conv_blocks=5, n_out_filters=160)
    self.fc = nn.Linear(in_features=int(linear_dim), out_features=n_classes)

  def forward(self, x):
    return self.fc(self.flatten(self.conv_blocks(x.float()/255.)))

  def conv_block(self, start_channels, hidden_channels=10, out_channels=10,
                 kernel_size=3, conv_stride=1, conv_padding=1,
                 pool_kernel_size=2, pool_stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=start_channels,
                  out_channels=hidden_channels,
                  kernel_size=kernel_size,
                  stride=conv_stride,
                  padding=conv_padding),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=conv_stride,
                  padding=conv_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=pool_kernel_size,
                     stride=pool_stride)
    )

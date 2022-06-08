import torch
import torch.nn as nn
import torchvision

class Mobilenet_v2_LK(torchvision.models.MobileNetV2):

	def __init__(self, in_channels=3, num_classes=136):
		super(Mobilenet_v2_LK, self).__init__(num_classes=num_classes)
		self.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=(3,3),
            bias=False,
            )
		nn.init.kaiming_normal_(self.features[0][0].weight, mode='fan_out')

	def forward(self, x):
		x = self.features(x)
		x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x
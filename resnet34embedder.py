import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """For now we're using resnet34, so we're using basic block, expansion=1 throughout."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet34Embedder(nn.Module):
    """
    ResNet-34 backbone (from primitives) that outputs a 512-D vector.
    Stages: [3, 4, 6, 3] basic blocks with channels [64, 128, 256, 512].
    Input expected: (B, 3, 128, 128).
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.inplanes = 64

        # Stem: 7x7 conv stride 2 + 3x3 maxpool stride 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(self.inplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-34 layers
        self.layer1 = self._make_layer(BasicBlock, 64,  3, stride=1)  # output: 32x32
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)  # output: 16x16
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)  # output: 8x8
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)  # output: 4x4

        # Global Average Pool → 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head: 512 → 512 (TODO: should we add an MLP later?)
        self.proj = nn.Linear(512 * BasicBlock.expansion, embed_dim, bias=True)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create one ResNet stage made of `blocks` BasicBlocks."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        # Kaiming init for convs; BN to sane defaults; proj to small std.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Expect x: (B, 3, 128, 128); channels-last is optional outside.
        x = self.conv1(x)     # -> (B, 64, 64, 64)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # -> (B, 64, 32, 32)

        x = self.layer1(x)    # -> (B, 64, 32, 32)
        x = self.layer2(x)    # -> (B, 128, 16, 16)
        x = self.layer3(x)    # -> (B, 256, 8, 8)
        x = self.layer4(x)    # -> (B, 512, 4, 4)

        x = self.avgpool(x)   # -> (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # -> (B, 512)
        x = self.proj(x)      # -> (B, 512)
        return x


# ---- quick smoke test ----
if __name__ == "__main__":
    model = ResNet34Embedder(embed_dim=512)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, 3, 128, 128)  # (B, C, H, W)
        out = model(dummy)
        print("Output shape:", out.shape)  # torch.Size([2, 512])
        print("L2 norms:", out.norm(dim=-1))
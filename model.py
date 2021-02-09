import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, a=1e-2):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
    nn.init.kaiming_uniform_(conv.weight, a=a)
    relu = nn.LeakyReLU(negative_slope=a)
    return nn.Sequential(conv, relu)


def fc(in_dim, out_dim, a=1e-2):
    lin = nn.Linear(in_dim, out_dim)
    if a < 0:
        nn.init.uniform_(lin.weight)
        return lin
    nn.init.kaiming_uniform_(lin.weight, a=a)
    relu = nn.LeakyReLU(negative_slope=a)
    return nn.Sequential(lin, relu)


class Encoder(nn.Module):
    """ Accepts a "PUV" (proposal, optical flow) state as input, and produces
        high level features.
    """

    def __init__(self):
        super().__init__()

        self.conv1 = conv3x3(3, 8)    # 254
        self.conv2 = conv3x3(8, 8)    # 252
        self.pool1 = nn.MaxPool2d(2)  # 126

        self.conv3 = conv3x3(8, 16)   # 124
        self.conv4 = conv3x3(16, 16)  # 122
        self.conv5 = conv3x3(16, 16)  # 120
        self.pool2 = nn.MaxPool2d(2)  # 60

        self.conv6 = conv3x3(16, 32)  # 58
        self.conv7 = conv3x3(32, 32)  # 56
        self.pool3 = nn.MaxPool2d(2)  # 28

        self.conv8 = conv3x3(32, 64)  # 26
        self.conv9 = conv3x3(64, 64)  # 24
        self.pool4 = nn.MaxPool2d(2)  # 12

    def forward(self, state):
        x = state['flow_features']
        vf = state['visual_features']

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.pool4(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, vf], dim=1)

        return x


class SelectionNetwork(nn.Module):
    """ Decides if a mask is moving/not moving.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fc1 = fc(64 * 12 * 12 + 256, 4096)
        self.fc2 = fc(4096, 2048)
        self.fc3 = fc(2048, 1024)
        self.fc4 = fc(1024, 512)
        self.fc5 = fc(512, 2, a=-1.)

    def forward(self, state):
        x = self.encoder(state)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


class AssignmentNetwork(nn.Module):
    """ Decides if two masks are moving together or not.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fc1 = fc((64 * 12 * 12 + 256) * 2, 4096)
        self.fc2 = fc(4096, 2048)
        self.fc3 = fc(2048, 1024)
        self.fc4 = fc(1024, 512)
        self.fc5 = fc(512, 2, a=-1.)

    def forward(self, states):
        state1, state2 = states
        # Sharing weights in a siamese fashion.
        x1 = self.encoder(state1)
        x2 = self.encoder(state2)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

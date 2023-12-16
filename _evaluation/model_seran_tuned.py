from torch import nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)  # expand_as y를 x의 크기로 확장


class Residual_Attention_Net1(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.channel1 = 2*planes
        self.channel2 = 2*self.channel1

        self.conv1 = conv3x3(inplanes, planes)

        self.downsample1 = nn.Conv2d(
            planes, self.channel1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.downsample2 = nn.Conv2d(
            self.channel1, self.channel2, kernel_size=2, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channel2)
        self.upsample1 = nn.ConvTranspose2d(
            self.channel2, self.channel1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channel1)
        self.upsample2 = nn.ConvTranspose2d(
            self.channel1, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        return residual_unit(self._softmask(x))*self._trunk(x)

    def _softmask(self, x):
        out = self.downsample1(x)
        out = self.bn1(out)
        out = self.downsample2(out)
        out = self.bn2(out)
        out = self.upsample1(out)
        out = self.bn3(out)
        out = self.upsample2(out)
        softmask_branch = self.softmax(out)
        return softmask_branch

    def _trunk(self, x):
        trunk_branch = self.conv1(x)
        return trunk_branch


class Residual_Attention_Net2(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.channel1 = 2*planes
        self.channel2 = 2*self.channel1

        self.conv2 = conv3x3(planes, planes)

        self.downsample1 = nn.Conv2d(
            planes, self.channel1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.downsample2 = nn.Conv2d(
            self.channel1, self.channel2, kernel_size=2, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.channel2)
        self.upsample1 = nn.ConvTranspose2d(
            self.channel2, self.channel1, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channel1)
        self.upsample2 = nn.ConvTranspose2d(
            self.channel1, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        return residual_unit(self._softmask(x))*self._trunk(x)

    def _softmask(self, x):
        out = self.downsample1(x)
        out = self.bn1(out)
        out = self.downsample2(out)
        out = self.bn2(out)
        out = self.upsample1(out)
        out = self.bn3(out)
        out = self.upsample2(out)
        softmask_branch = self.softmax(out)
        return softmask_branch

    def _trunk(self, x):
        trunk_branch = self.conv2(x)
        return trunk_branch


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3,
                     padding=1,
                     bias=False)


def residual_unit(soft_mask_branch):
    return torch.add(soft_mask_branch, 1)


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_module = SELayer(planes)
        self.attention1 = Residual_Attention_Net1(inplanes, planes)
        self.attention2 = Residual_Attention_Net2(planes)

    def forward(self, x):
        residual = x
        residual_unit1 = self.attention1(x)  # Residual module
        se_unit2 = self.se_module(residual_unit1)  # SE-layer
        out = self.bn1(se_unit2)
        out = self.relu(out)
        residual_unit2 = self.attention2(out)  # Residual module
        se_unit3 = self.se_module(residual_unit2)  # SE-layer
        out = self.bn2(se_unit3)
        out += residual
        out = self.relu(out)
        se_unit4 = self.se_module(out)  # SE-layer
        return se_unit4

class PolicyHead(nn.Module):
    def __init__(self, planes, board_size):
        super().__init__()
        self.policy_head = nn.Conv2d(planes, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(True)
        self.policy_fc = nn.Linear(board_size**2 * 2, board_size**2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x = (1,128,board_size,board_size)
        out = self.policy_head(x)
        # out = (1, 2, board_size, board_size)
        out = self.policy_bn(out)
        # out = (1, 2, board_size, board_size)
        out = self.relu(out)
        # out = (1, 2, board_size, board_size)
        out = out.view(out.size(0), -1)
        # out = (1, 450)
        out = self.policy_fc(out)
        # out = (1, board_size**2)
        out = self.softmax(out)
        # out = (1, board_size**2)
        return out

class ValueHead(nn.Module):
    def __init__(self, planes, board_size):
        super().__init__()
        self.value_head = nn.Conv2d(planes, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(True)
        self.value_fc1 = nn.Linear(board_size**2, planes)
        self.value_fc2 = nn.Linear(planes, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x = (1,128,board_size,board_size)
        out = self.value_head(x)
        # x = (1,1,board_size,board_size)
        out = self.value_bn(out)
        # x = (1,1,board_size,board_size)
        out = self.relu(out)
        # x = (1,1,board_size,board_size)
        out = out.view(out.size(0), -1)
        # x = (1, board_size**2)
        out = self.value_fc1(out)
        # x = (1, 128)
        out = self.relu(out)
        # x = (1, 128)
        out = self.value_fc2(out)
        # x = (1, 1)
        out = self.tanh(out)
        # x = (1, 1) => [[]]
        out = out.view(out.size(0))
        # x = (1,) => []
        return out


class PVNet(nn.Module):
    def __init__(self, n_block, inplanes, planes, board_size):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        self.layers = self._make_layer(ResBlock, planes, n_block)
        self.policy_head = PolicyHead(planes, board_size)
        self.value_head = ValueHead(planes, board_size)
        self.se_module = SELayer(planes)

        for m in self.modules(): 
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, Resblock, planes, n_block):
        blocks = []
        for i in range(n_block):
            blocks.append(Resblock(planes, planes))
        return nn.Sequential(*blocks)

    def forward(self, x):
        # x = (1, 5, board_size, board_size)
        x = self.conv1(x)
        # x = (1,128,board_size,board_size)
        se_unit1 = self.se_module(x)
        x = self.bn1(se_unit1)
        # x = (1,128,board_size,board_size)
        x = self.relu(x)
        # x = (1,128,board_size,board_size)
        x = self.layers(x)
        # x = (1,128,board_size,board_size)
        p = self.policy_head(x)
        # p = (1, board_size**2)
        v = self.value_head(x)
        # x = (1,) => []
        return p, v

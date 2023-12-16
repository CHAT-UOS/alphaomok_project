from torch import nn
import torch

class Residual_Attention_Net1(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.channel1 = 2*planes
        self.channel2 = 2*self.channel1
        self.conv1 = conv3x3(inplanes, planes)
        self.downsample1 = nn.MaxPool2d(2)
        self.downsample2 = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2.25, mode='bilinear')
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        return residual_unit(self._softmask(x))*self._trunk(x)

    def _softmask(self, x):
        out = self.downsample1(x)
        out = self.downsample2(out)
        out = self.upsample1(out)
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
        self.conv1 = conv3x3(planes, planes)
        self.downsample1 = nn.MaxPool2d(2)
        self.downsample2 = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2.25, mode='bilinear')
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        return residual_unit(self._softmask(x))*self._trunk(x)

    def _softmask(self, x):
        out = self.downsample1(x)
        out = self.downsample2(out)
        out = self.upsample1(out)
        out = self.upsample2(out)
        softmask_branch = self.softmax(out)
        return softmask_branch
    
    def _trunk(self, x):
        trunk_branch = self.conv1(x)
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
        self.bn2 = nn.BatchNorm2d(planes)
        self.attention1 = Residual_Attention_Net1(inplanes, planes)
        self.attention2 = Residual_Attention_Net2(planes)

    def forward(self, x):
        residual = x
        residual_unit1 = self.attention1(x)  
        out = self.bn1(residual_unit1)
        out = self.relu(out)
        out = self.attention2(out)
        out= self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

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
        x = self.bn1(x)
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

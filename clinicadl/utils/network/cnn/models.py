import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models.resnet import BasicBlock

from clinicadl.utils.network.cnn.resnet import ResNetDesigner, model_urls
from clinicadl.utils.network.cnn.resnet3D import ResNetDesigner3D
from clinicadl.utils.network.cnn.attentionnet import AttentionDesigner3D
from clinicadl.utils.network.cnn.SECNN import SECNNDesigner3D


from clinicadl.utils.network.network_utils import PadMaxPool2d, PadMaxPool3d
from clinicadl.utils.network.sub_network import CNN, GNet, CNN_da, CNN_SSDA


def get_layers_fn(input_size):
    if len(input_size) == 4:
        return nn.Conv3d, nn.BatchNorm3d, PadMaxPool3d
        # return nn.Conv3d, nn.GroupNorm, PadMaxPool3d

        # return nn.Conv3d, nn.InstanceNorm3d, PadMaxPool3d

    elif len(input_size) == 3:
        return nn.Conv2d, nn.BatchNorm2d, PadMaxPool2d
    else:
        raise ValueError(
            f"The input is neither a 2D or 3D image.\n "
            f"Input shape is {input_size - 1}."
        )


class Conv5_FC3(CNN):
    """
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),

            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),

            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),

            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),

            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )


class Conv4_FC3(CNN):
    """
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),

            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),

            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),

            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),

            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 50),
            nn.ReLU(),

            nn.Linear(50, 40),
            nn.ReLU(),

            nn.Linear(40, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )


class resnet18(CNN):
    def __init__(self, input_size, gpu=False, output_size=2, dropout=0.5):
        model = ResNetDesigner(input_size, BasicBlock, [2, 2, 2, 2])
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))

        convolutions = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )

        # add a fc layer on top of the transfer_learning model and a softmax classifier
        fc = nn.Sequential(nn.Flatten(), model.fc)
        fc.add_module("drop_out", nn.Dropout(p=dropout))
        fc.add_module("fc_out", nn.Linear(1000, output_size))

        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )


class ResNet3D(CNN):
    def __init__(
        self, input_size=[1, 169, 208, 179], gpu=False, output_size=2, dropout=0.5
    ):
        model = ResNetDesigner3D()

        convolutions = nn.Sequential(
            model.layer0, model.layer1, model.layer2, model.layer3, model.layer4
        )

        fc = model.fc

        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )


class AttentionNet(CNN):
    def __init__(
        self, input_size=[1, 169, 208, 179], gpu=False, output_size=2, dropout=0.5
    ):
        model = AttentionDesigner3D()

        convolutions = nn.Sequential(
            model.pre_conv,
            model.stage1,
            model.stage2,
            model.stage3,
            model.stage4,
            model.avg,
        )

        fc = model.classifier

        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )


# class GoogLeNet3D(CNN):
#     def __init__(self):
#         model = GoogLeNet3D_Designer()
#         convolution = nn.Sequential(
#             model.pre_layers,
#             model.a3,
#             model.b3,
#             model.maxpool,
#             model.a4
#         )

#         # For gradient injection
#         if self.training:
#             aux_out1 = self.aux1(out)

#         out = self.b4(out)
#         out = self.c4(out)
#         out = self.d4(out)

#         # For gradient injection
#         if self.training:
#             aux_out2 = self.aux2(out)

#         out = self.e4(out)
#         out = self.maxpool(out)
#         out = self.a5(out)
#         out = self.b5(out)
#         out = self.avgpool(out)
#         out = F.dropout(out, 0.4, training=self.training)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         if self.training:
#             return out, aux_out1, aux_out2
#         return out


class Stride_Conv5_FC3(CNN):
    """
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1, stride=2),
            norm(8),
            nn.ReLU(),

            conv(8, 16, 3, padding=1, stride=2),
            norm(16),
            nn.ReLU(),

            conv(16, 32, 3, padding=1, stride=2),
            norm(32),
            nn.ReLU(),

            conv(32, 64, 3, padding=1, stride=2),
            norm(64),
            nn.ReLU(),

            conv(64, 128, 3, padding=1, stride=2),
            norm(128),
            nn.ReLU(),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )


class SE_CNN(CNN):
    def __init__(
        self, input_size=[1, 169, 208, 179], gpu=True, output_size=2, dropout=0.5
    ):
        model = SECNNDesigner3D()

        convolutions = nn.Sequential(
            model.layer0, model.layer1, model.layer2, model.layer3, model.layer4
        )

        fc = model.fc

        super().__init__(
            convolutions=convolutions,
            fc=fc,
            n_classes=output_size,
            gpu=gpu,
        )


class Gnet_Conv5_FC3(GNet):
    """
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        conv3d_c1_border = nn.Sequential(conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2))
        conv3d_c1_center = nn.Sequential(conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2))

        convolutions = nn.Sequential(
            # conv(input_size[0], 8, 3, padding=1),
            # norm(8),
            # nn.ReLU(),
            # pool(2, 2),

            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),

            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),

            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),

            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros([8, 85, 85, 90]).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc=fc,
            conv3d_c1_center=conv3d_c1_center,
            conv3d_c1_border=conv3d_c1_border,
            n_classes=output_size,
            gpu=gpu,
        )


class Conv5_FC3_inv(CNN_da):
    """
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),

            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),

            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),

            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),

            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),

            # conv(128, 256, 3, padding=1),
            # norm(256),
            # nn.ReLU(),
            # pool(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc_class = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )

        fc_domain = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc_class=fc_class,
            fc_domain=fc_domain,
            n_classes=output_size,
            gpu=gpu,
        )


class Conv5_FC3_MME(CNN_SSDA):
    """
    It is a convolutional neural network with 5 convolution and 3 fully-connected layer.
    It reduces the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, gpu=True, output_size=2, dropout=0.5):
        conv, norm, pool = get_layers_fn(input_size)
        # fmt: off
        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),

            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),

            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),

            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),

            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
        )

        fc_c = nn.Sequential(
            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions,
            fc=fc,
            fc_c=fc_c,
            n_classes=output_size,
            gpu=gpu,
        )

    @staticmethod
    def get_input_size():
        return "1@128x128"

    @staticmethod
    def get_dimension():
        return "2D or 3D"

    @staticmethod
    def get_task():
        return ["classification", "regression"]

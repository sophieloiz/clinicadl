from collections import OrderedDict
from logging import getLogger

import torch
from torch import nn

from clinicadl.utils.exceptions import ClinicaDLNetworksError
from clinicadl.utils.network.network import Network
from clinicadl.utils.network.network_utils import (
    CropMaxUnpool2d,
    CropMaxUnpool3d,
    PadMaxPool2d,
    PadMaxPool3d,
    GradReverse,
)
from torch.fft import fftn, ifftn, fftshift, ifftshift
from torch.autograd import Function
import torch.nn.functional as F

logger = getLogger("clinicadl")


class AutoEncoder(Network):
    def __init__(self, encoder, decoder, gpu=False):
        super().__init__(gpu=gpu)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    @property
    def layers(self):
        return nn.Sequential(self.encoder, self.decoder)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, AutoEncoder):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, CNN):
            encoder_dict = OrderedDict(
                [
                    (k.replace("convolutions.", ""), v)
                    for k, v in state_dict.items()
                    if "convolutions" in k
                ]
            )
            self.encoder.load_state_dict(encoder_dict)
        else:
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to Autoencoder."
            )

    def predict(self, x):
        _, output = self.forward(x)
        return output

    def forward(self, x):
        indices_list = []
        pad_list = []
        for layer in self.encoder:
            if (
                (isinstance(layer, PadMaxPool3d) or isinstance(layer, PadMaxPool2d))
                and layer.return_indices
                and layer.return_pad
            ):
                x, indices, pad = layer(x)
                indices_list.append(indices)
                pad_list.append(pad)
            elif (
                isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.MaxPool2d)
            ) and layer.return_indices:
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)

        code = x.clone()

        for layer in self.decoder:
            if isinstance(layer, CropMaxUnpool3d) or isinstance(layer, CropMaxUnpool2d):
                x = layer(x, indices_list.pop(), pad_list.pop())
            elif isinstance(layer, nn.MaxUnpool3d) or isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)

        return code, x

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images = input_dict["image"].to(self.device)
        train_output = self.predict(images)
        loss = criterion(train_output, images)

        return train_output, {"loss": loss}


class CNN(Network):
    def __init__(self, convolutions, fc, n_classes, gpu=False):
        super().__init__(gpu=gpu)
        self.convolutions = convolutions.to(self.device)
        self.fc = fc.to(self.device)
        self.n_classes = n_classes

    @property
    def layers(self):
        return nn.Sequential(self.convolutions, self.fc)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict(
                [
                    (k.replace("encoder.", ""), v)
                    for k, v in state_dict.items()
                    if "encoder" in k
                ]
            )
            self.convolutions.load_state_dict(convolutions_dict)
        else:
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to CNN."
            )

    def forward(self, x):
        x = self.convolutions(x)
        # print("Attention Net : resize function")
        # x = x.view(x.size(0), -1) # for attentionnet
        return self.fc(x)

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output = self.forward(images)
        if use_labels:
            loss = criterion(train_output, labels)
        else:
            loss = torch.Tensor([0])

        return train_output, {"loss": loss}


class ViT(Network):
    def __init__(
        self,
        patch_to_embedding,
        transformer,
        to_latent,
        mlp_head,
        dropout,
        pos_embedding,
        cls_token,
        patch_size,
        pool="cls",
        gpu=True,
    ):
        super().__init__(gpu=True)
        self.patch_to_embedding = patch_to_embedding.to(self.device)
        self.transformer = transformer.to(self.device)
        self.to_latent = to_latent.to(self.device)
        self.mlp_head = mlp_head.to(self.device)
        self.pos_embedding = pos_embedding.to(self.device)
        self.cls_token = cls_token.to(self.device)
        self.patch_size = patch_size
        self.dropout = dropout
        self.pool = pool

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN):
            self.load_state_dict(state_dict)
        if issubclass(transfer_class, ViT):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict(
                [
                    (k.replace("encoder.", ""), v)
                    for k, v in state_dict.items()
                    if "encoder" in k
                ]
            )
            self.convolutions.load_state_dict(convolutions_dict)
        else:
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to class ViTVNet:."
            )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(
            img[:, :, :, :, :176],
            "b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)",
            p1=p,
            p2=p,
            p3=16,
        )
        print(x.shape)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output = self.forward(images)
        if use_labels:
            loss = criterion(train_output, labels)
        else:
            loss = torch.Tensor([0])

        return train_output, {"loss": loss}


class GNet(Network):
    def __init__(
        self,
        convolutions,
        fc,
        conv3d_c1_border,
        conv3d_c1_center,
        n_classes,
        gpu=False,
    ):
        super().__init__(gpu=gpu)
        self.convolutions = convolutions.to(self.device)
        self.fc = fc.to(self.device)
        self.conv3d_c1_border = conv3d_c1_border.to(self.device)
        self.conv3d_c1_center = conv3d_c1_center.to(self.device)
        self.n_classes = n_classes

    @property
    def layers(self):
        return nn.Sequential(self.convolutions, self.fc)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, GNet):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict(
                [
                    (k.replace("encoder.", ""), v)
                    for k, v in state_dict.items()
                    if "encoder" in k
                ]
            )
            self.convolutions.load_state_dict(convolutions_dict)
        else:
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to CNN."
            )

    # For GNet model: fusion operation
    def merge_center_border(self, tensor_center, tensor_border, beta=0.1):
        # Get the image size and do the localization to get the center
        width = tensor_border.size()[2]
        mask_width = int(width * beta)
        mask_star_x = int(width / 2 - mask_width / 2)
        mask_star_y = mask_star_x
        mask_end_x = int(width / 2 + mask_width / 2)
        mask_end_y = mask_end_x
        # Set the value of center-frequency to the center
        tensor_border[
            :, :, mask_star_x:mask_end_x, mask_star_y:mask_end_y, :
        ] = tensor_center
        return tensor_border

    def mask_tensor(input_tensor, beta=0.1):
        batch = input_tensor.size()[0]
        classes = input_tensor.size()[1]
        width = input_tensor.size()[2]
        slice = input_tensor.size()[4]
        mask_width = int(width * beta)
        mask_star_x = int(width / 2 - mask_width / 2)
        mask_star_y = mask_star_x
        mask_end_x = int(width / 2 + mask_width / 2)
        mask_end_y = mask_end_x
        mask = torch.zeros((batch, classes, width, width, slice))
        mask[:, :, mask_star_x:mask_end_x, mask_star_y:mask_end_y, :] = torch.ones(
            (batch, classes, mask_width, mask_width, slice)
        )
        input_fft_center = input_tensor[
            :, :, mask_star_x:mask_end_x, mask_star_y:mask_end_y, :
        ]

        input_fft_border = torch.multiply(input_tensor, 1 - mask)
        return input_fft_center, input_fft_border

    def mask_tensor(self, input_tensor, beta=0.1):
        batch = input_tensor.size()[0]
        classes = input_tensor.size()[1]
        width = input_tensor.size()[2]
        slice = input_tensor.size()[4]
        mask_width = int(width * beta)
        mask_star_x = int(width / 2 - mask_width / 2)
        mask_star_y = mask_star_x
        mask_end_x = int(width / 2 + mask_width / 2)
        mask_end_y = mask_end_x
        mask = torch.zeros((batch, classes, width, width, slice)).to(self.device)
        mask[:, :, mask_star_x:mask_end_x, mask_star_y:mask_end_y, :] = torch.ones(
            (batch, classes, mask_width, mask_width, slice)
        )
        input_fft_center = input_tensor[
            :, :, mask_star_x:mask_end_x, mask_star_y:mask_end_y, :
        ]
        print("Input tensor shape")
        print(input_tensor.shape)
        print(mask.shape)

        input_fft_border = torch.multiply(input_tensor, 1 - mask)
        return input_fft_center, input_fft_border

    def forward(self, x):
        # x = self.convolutions(x)
        # out_classifier = self.fc_classifier(x)
        # out_domain = self.fc_domain(x)

        # input_tensor, target = torch.zeros((160,160,128)), torch.zeros((160,160,128))
        print(x.shape)
        x = x[:, :, :169, :169, :]
        print(x.shape)
        input_dim = tuple(range(2, x.ndim))
        print("Distangle")
        # Do fftn for input to disentangle
        input_f = fftn(x, dim=input_dim)
        # input_f = fftshift(
        #   input_f
        # )  # [ONLY-LOW] For the high-frequency model, we just remove this line
        # Use mask to get the center and border
        print("Mask")
        input_center_f, input_border_f = self.mask_tensor(input_f)
        # input_center_f = ifftshift(
        #   input_center_f
        # )  # [ONLY-LOW] For the high-frequency model, we just remove this line
        # input_border_f = ifftshift(
        #   input_border_f
        # )  # [ONLY-LOW] For the high-frequency model, we just remove this line
        input_center_i = ifftn(
            input_center_f, dim=tuple(range(2, input_center_f.ndim))
        ).type(torch.float32)
        input_border_i = ifftn(
            input_border_f, dim=tuple(range(2, input_border_f.ndim))
        ).type(torch.float32)
        input_center_i, input_border_i = input_center_i.cuda(), input_border_i.cuda()
        input_center_i.requires_grad = True
        input_border_i.requires_grad = True

        # use CNN to process x_border_i (means border frequency part of input x in image space) and x_center_i (means center frequency part of input x in image space)
        output_border_i = self.conv3d_c1_border(
            input_border_i
        )  # 1 Conv3D of the conv5fc3
        output_center_i = self.conv3d_c1_center(input_center_i)  # 1 Conv3D
        # do fftn to transfer output (output_border_f and output_center_f) transform from image to frequency domain
        output_border_f = fftn(
            output_border_i, dim=tuple(range(2, output_border_i.ndim))
        )
        # output_border_f = fftshift(
        #   output_border_f
        # )  # [ONLY-LOW] For the high-frequency model, we just remove this line
        output_center_f = fftn(
            output_center_i, dim=tuple(range(2, output_center_i.ndim))
        )
        # output_center_f = fftshift(
        #   output_center_f
        # )  # [ONLY-LOW] For the high-frequency model, we just remove this line
        # merge output of border and center frequency together in Frequency domain
        output_center_border_f = self.merge_center_border(
            output_center_f, output_border_f
        )
        # Transform back to image domain
        # output_center_border_f = ifftshift(
        #   output_center_border_f
        # )  # [ONLY-LOW] For the high-frequency model, we just remove this line
        output_center_border_i = ifftn(
            output_center_border_f, dim=tuple(range(2, output_center_border_f.ndim))
        ).type(torch.float32)
        out = output_center_border_i
        print("Shape out")
        print(out.shape)
        # Do downstream task conv5fc3
        x = self.convolutions(out)
        out_classifier = self.fc(x)

        return out_classifier

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output = self.forward(images)
        if use_labels:
            loss = criterion(train_output, labels)
        else:
            loss = torch.Tensor([0])

        return train_output, {"loss": loss}


class ReverseLayerF(Function):
    def forward(self, x, alpha):
        self.alpha = alpha
        return x.view_as(x)

    def backward(self, grad_output):
        output = grad_output.neg() * self.alpha

        return output, None


class CNN_DANN(Network):
    def __init__(self, convolutions, fc_class, fc_domain, n_classes, gpu=False):  #
        super().__init__(gpu=gpu)
        self.convolutions = convolutions.to(self.device)
        self.fc_class = fc_class.to(self.device)
        self.fc_domain = fc_domain.to(self.device)
        self.n_classes = n_classes

    @property
    def layers(self):
        return nn.Sequential(self.convolutions, self.fc_class, self.fc_domain)  # ,

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN_DANN):
            self.load_state_dict(state_dict)

        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict(
                [
                    (k.replace("encoder.", ""), v)
                    for k, v in state_dict.items()
                    if "encoder" in k
                ]
            )
            self.convolutions.load_state_dict(convolutions_dict)
        else:
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to CNN."
            )

    def grad_reverse(self, x, lambd=1.0):
        return GradReverse(lambd)(x)

    def forward(self, x, alpha):
        x = self.convolutions(x)
        x_class = self.fc_class(x)
        x_reverse = ReverseLayerF.apply(x, alpha)
        x_domain = self.fc_domain(x_reverse)
        return x_class, x_domain

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss_new_lab(
        self, data_lab, data_target_unl, criterion, alpha
    ):

        images, labels, domain = (
            data_lab["image"].to(self.device),
            data_lab["label"].to(self.device),
            data_lab["domain"],  # .to(self.device),
        )

        logger.info(f"Label : {labels}")

        images_target_unl = data_target_unl["image"].to(self.device)

        train_output_class, train_output_domain = self.forward(images, alpha)

        _, train_output_domain_target_lab = self.forward(images_target_unl, alpha)

        loss_classif = criterion(train_output_class, labels)

        output_array_domain = [0 if element == "t1" else 1 for element in domain]

        output_tensor_domain = torch.tensor(output_array_domain).to(self.device)

        logger.info(f"domain : {output_array_domain}")

        labels_domain_t = (
            torch.ones(data_target_unl["image"].shape[0]).long().to(self.device)
        )

        loss_domain_lab = criterion(train_output_domain, output_tensor_domain)
        loss_domain_t_unl = criterion(train_output_domain_target_lab, labels_domain_t)

        loss_domain = loss_domain_lab + loss_domain_t_unl

        total_loss = loss_classif + loss_domain

        return (
            train_output_class,
            train_output_domain,
            {"loss": total_loss},
        )

    def compute_outputs_and_loss_new(
        self, input_dict, input_dict_target, input_dict_target_unl, criterion, alpha
    ):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        logger.info(labels)
        images_target, labels_target = input_dict_target["image"].to(
            self.device
        ), input_dict_target["label"].to(self.device)

        images_target_unl = input_dict_target_unl["image"].to(self.device)

        train_output_class_source, train_output_domain_source = self.forward(
            images, alpha
        )
        train_output_class_target, train_output_domain_target = self.forward(
            images_target, alpha
        )
        _, train_output_domain_target_lab = self.forward(images_target_unl, alpha)

        loss_source = criterion(train_output_class_source, labels)
        loss_target = criterion(train_output_class_target, labels_target)

        loss_classif = loss_source + loss_target

        labels_domain_s = (
            torch.zeros(input_dict["image"].shape[0]).long().to(self.device)
        )

        labels_domain_t = (
            torch.ones(input_dict_target["image"].shape[0]).long().to(self.device)
        )

        loss_domain_s = criterion(train_output_domain_source, labels_domain_s)
        loss_domain_t = criterion(train_output_domain_target, labels_domain_t)
        loss_domain_t_unl = criterion(train_output_domain_target_lab, labels_domain_t)

        loss_domain = loss_domain_s + loss_domain_t + loss_domain_t_unl

        total_loss = loss_classif + loss_domain

        return (
            train_output_class_source,
            train_output_class_target,
            {"loss": total_loss},
        )

    def compute_outputs_and_loss_domain(
        self, input_dict, input_dict_target, criterion, alpha, use_labels=True
    ):

        images = input_dict["image"].to(self.device)

        images_target = input_dict_target["image"].to(self.device)

        _, train_output_domain_source = self.forward(images, alpha)
        _, train_output_domain_target = self.forward(images_target, alpha)

        labels_domain_s = (
            torch.zeros(input_dict["image"].shape[0]).long().to(self.device)
        )

        labels_domain_t = (
            torch.ones(input_dict_target["image"].shape[0]).long().to(self.device)
        )

        loss_domain_s = criterion(train_output_domain_source, labels_domain_s)
        loss_domain_t = criterion(train_output_domain_target, labels_domain_t)
        loss_domain = loss_domain_s + loss_domain_t

        return (
            train_output_domain_source,
            train_output_domain_target,
            {"loss_domain": loss_domain},
        )

    def compute_outputs_and_loss(self, input_dict, criterion, alpha=0, use_labels=True):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )

        train_output_class, _ = self.forward(images, alpha)

        if use_labels:
            loss = criterion(train_output_class, labels)
        else:
            loss = torch.Tensor([0])

        return train_output_class, {"loss": loss}

    def compute_outputs_and_loss_test(self, input_dict, criterion, alpha, target):
        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output, train_output_domain = self.forward(images, alpha)

        loss_bce = criterion(train_output, labels)
        if target:
            labels_domain_t = (
                torch.ones(input_dict["image"].shape[0]).long().to(self.device)
            )
        else:
            labels_domain_t = (
                torch.zeros(input_dict["image"].shape[0]).long().to(self.device)
            )
        loss_domain = criterion(train_output_domain, labels_domain_t)

        return train_output, {"loss": loss_bce + alpha * loss_domain}

    # Define the learning rate scheduler function
    def lr_scheduler(self, lr, optimizer, p):
        lr = lr / (1 + 10 * p) ** 0.75
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return optimizer


class CNN_MME(Network):
    def __init__(self, convolutions, fc, fc_c, n_classes, gpu=False):
        super().__init__(gpu=gpu)
        self.convolutions = convolutions.to(self.device)
        self.fc = fc.to(self.device)
        self.fc_c = fc_c.to(self.device)
        self.n_classes = n_classes

    @property
    def layers(self):
        return nn.Sequential(self.convolutions, self.fc, self.fc_c)

    def grad_reverse(self, x, lambd=1.0):
        return GradReverse(lambd)(x)

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, CNN_MME):
            self.load_state_dict(state_dict)
        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict(
                [
                    (k.replace("encoder.", ""), v)
                    for k, v in state_dict.items()
                    if "encoder" in k
                ]
            )
            self.convolutions.load_state_dict(convolutions_dict)
        else:
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to CNN."
            )

    def forward(self, x, reverse=False, eta=0.1, temp=0.05):
        x = self.convolutions(x)
        out_g = self.fc(x)
        if reverse:
            out_g = ReverseLayerF.apply(out_g, eta)

        out_c = F.normalize(out_g)
        out_c = self.fc_c(out_c) / temp
        return out_g, out_c

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss_bce(self, input_dict, criterion, use_labels=True):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        _, train_output = self.forward(images)
        if use_labels:
            loss = criterion(train_output, labels)
        else:
            loss = torch.Tensor([0])

        return train_output, {"loss_bce": loss}

    def compute_outputs_and_loss(self, input_dict, criterion, use_labels=True):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        _, train_output = self.forward(images)
        if use_labels:
            loss = criterion(train_output, labels)
        else:
            loss = torch.Tensor([0])

        return train_output, {"loss": loss}

    def adentropy(self, train_output, lamda=0.1, eta=1.0):
        out_t1 = F.softmax(train_output)
        loss_adent = lamda * torch.mean(
            torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1)
        )
        return loss_adent

    # For ENT training (need to create a new class with loss with add entropy)
    def entropy(self, train_output, lamda=0.1, eta=1.0):
        out_t1 = F.softmax(train_output)
        loss_adent = -lamda * torch.mean(
            torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1)
        )
        return loss_adent

    def compute_outputs_and_loss_entropy(self, input_dict):

        images = input_dict["image"].to(self.device).to(self.device)
        _, train_output = self.forward(images, reverse=True)
        loss = self.adentropy(train_output)
        # loss = self.entropy(train_output)

        return train_output, {"loss_entropy": loss}

    def compute_outputs_and_loss_test(
        self, input_dict, criterion, alpha=0, target=None
    ):
        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        _, train_output = self.forward(images)
        _, train_output_ = self.forward(images, reverse=True)
        loss_bce = criterion(train_output, labels)
        loss_a = self.adentropy(train_output_)

        return train_output, {"loss": loss_bce + loss_a}

    def inv_lr_scheduler(
        self, param_lr, optimizer, iter_num, gamma=0.0001, power=0.75, init_lr=0.001
    ):
        lr = init_lr * (1 + gamma * iter_num) ** (-power)
        i = 0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * param_lr[i]
            i += 1
        return optimizer


class CNN_DANN2ouputs(Network):
    def __init__(
        self,
        convolutions,
        fc_class_source,
        fc_class_target,
        fc_domain,
        n_classes,
        gpu=False,
    ):  #
        super().__init__(gpu=gpu)
        self.convolutions = convolutions.to(self.device)
        self.fc_class_source = fc_class_source.to(self.device)
        self.fc_class_target = fc_class_target.to(self.device)
        self.fc_domain = fc_domain.to(self.device)
        self.n_classes = n_classes

    @property
    def layers(self):
        return nn.Sequential(
            self.convolutions,
            self.fc_class_source,
            self.fc_class_target,
            self.fc_domain,
        )  # ,

    def transfer_weights(self, state_dict, transfer_class):
        if issubclass(transfer_class, CNN_DANN):
            self.load_state_dict(state_dict)

        elif issubclass(transfer_class, AutoEncoder):
            convolutions_dict = OrderedDict(
                [
                    (k.replace("encoder.", ""), v)
                    for k, v in state_dict.items()
                    if "encoder" in k
                ]
            )
            self.convolutions.load_state_dict(convolutions_dict)
        else:
            raise ClinicaDLNetworksError(
                f"Cannot transfer weights from {transfer_class} to CNN."
            )

    def grad_reverse(self, x, lambd=1.0):
        return GradReverse(lambd)(x)

    def forward(self, x, alpha):
        x = self.convolutions(x)
        x_class_source = self.fc_class_source(x)
        x_class_target = self.fc_class_target(x)
        x_reverse = ReverseLayerF.apply(x, alpha)
        x_domain = self.fc_domain(x_reverse)
        return x_class_source, x_class_target, x_domain

    def predict(self, x):
        return self.forward(x)

    def compute_outputs_and_loss_new_lab(
        self, data_lab, data_target_unl, criterion, alpha
    ):

        images, labels, domain = (
            data_lab["image"].to(self.device),
            data_lab["label"].to(self.device),
            data_lab["domain"],  # .to(self.device),
        )

        # Create binary flags for the domain values
        # Create binary flags for the domain values
        source_domain_flag = domain == "t1"  # 1 if domain is "t1", 0 otherwise
        target_domain_flag = domain == "flair"

        logger.info(f"Images shape {images.size()}")
        logger.info(f"Images shape {source_domain_flag}")
        logger.info(f"Images shape {target_domain_flag}")

        # Expand the dimensions of the domain flags to match the shape of the images tensor
        source_domain_flag = source_domain_flag.expand(
            -1, images.size(1), images.size(2)
        )
        target_domain_flag = target_domain_flag.expand(
            -1, images.size(1), images.size(2)
        )

        # Create copies of the image tensor based on the domain flags
        source_image_tensor = images * source_domain_flag
        target_image_tensor = images * target_domain_flag

        logger.info(f"Source images shape {source_image_tensor.size()}")
        logger.info(f"Target images shape {target_image_tensor.size()}")

        print(source_image_tensor)
        print(target_image_tensor)

        source_labels_tensor = labels * source_domain_flag
        target_label_tensor = labels * target_domain_flag

        logger.info(f"Source label shape {source_labels_tensor.size()}")
        logger.info(f"Target label shape {target_label_tensor.size()}")

        logger.info(f"Label : {labels}")

        images_target_unl = data_target_unl["image"].to(self.device)

        train_output_class_source, _, train_output_domain = self.forward(
            source_image_tensor, alpha
        )
        _, train_output_class_target, train_output_domain = self.forward(
            target_image_tensor, alpha
        )

        _, _, train_output_domain_target_lab = self.forward(images_target_unl, alpha)

        loss_classif_source = criterion(train_output_class_source, source_labels_tensor)
        loss_classif_target = criterion(train_output_class_target, target_label_tensor)

        loss_classif = loss_classif_source + loss_classif_target

        output_array_domain = [0 if element == "t1" else 1 for element in domain]

        output_tensor_domain = torch.tensor(output_array_domain).to(self.device)

        logger.info(f"domain : {output_array_domain}")

        labels_domain_t = (
            torch.ones(data_target_unl["image"].shape[0]).long().to(self.device)
        )

        loss_domain_lab = criterion(train_output_domain, output_tensor_domain)
        loss_domain_t_unl = criterion(train_output_domain_target_lab, labels_domain_t)

        loss_domain = loss_domain_lab + loss_domain_t_unl

        total_loss = loss_classif + loss_domain

        return (
            train_output_class,
            train_output_domain,
            {"loss": total_loss},
        )

    def compute_outputs_and_loss_new(
        self, input_dict, input_dict_target, input_dict_target_unl, criterion, alpha
    ):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        logger.info(labels)
        images_target, labels_target = input_dict_target["image"].to(
            self.device
        ), input_dict_target["label"].to(self.device)

        images_target_unl = input_dict_target_unl["image"].to(self.device)

        train_output_class_source, train_output_domain_source = self.forward(
            images, alpha
        )
        train_output_class_target, train_output_domain_target = self.forward(
            images_target, alpha
        )
        _, train_output_domain_target_lab = self.forward(images_target_unl, alpha)

        loss_source = criterion(train_output_class_source, labels)
        loss_target = criterion(train_output_class_target, labels_target)

        loss_classif = loss_source + loss_target

        labels_domain_s = (
            torch.zeros(input_dict["image"].shape[0]).long().to(self.device)
        )

        labels_domain_t = (
            torch.ones(input_dict_target["image"].shape[0]).long().to(self.device)
        )

        loss_domain_s = criterion(train_output_domain_source, labels_domain_s)
        loss_domain_t = criterion(train_output_domain_target, labels_domain_t)
        loss_domain_t_unl = criterion(train_output_domain_target_lab, labels_domain_t)

        loss_domain = loss_domain_s + loss_domain_t + loss_domain_t_unl

        total_loss = loss_classif + loss_domain

        return (
            train_output_class_source,
            train_output_class_target,
            {"loss": total_loss},
        )

    def compute_outputs_and_loss_domain(
        self, input_dict, input_dict_target, criterion, alpha, use_labels=True
    ):

        images = input_dict["image"].to(self.device)

        images_target = input_dict_target["image"].to(self.device)

        _, train_output_domain_source = self.forward(images, alpha)
        _, train_output_domain_target = self.forward(images_target, alpha)

        labels_domain_s = (
            torch.zeros(input_dict["image"].shape[0]).long().to(self.device)
        )

        labels_domain_t = (
            torch.ones(input_dict_target["image"].shape[0]).long().to(self.device)
        )

        loss_domain_s = criterion(train_output_domain_source, labels_domain_s)
        loss_domain_t = criterion(train_output_domain_target, labels_domain_t)
        loss_domain = loss_domain_s + loss_domain_t

        return (
            train_output_domain_source,
            train_output_domain_target,
            {"loss_domain": loss_domain},
        )

    def compute_outputs_and_loss(self, input_dict, criterion, alpha=0, use_labels=True):

        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )

        train_output_class, _ = self.forward(images, alpha)

        if use_labels:
            loss = criterion(train_output_class, labels)
        else:
            loss = torch.Tensor([0])

        return train_output_class, {"loss": loss}

    def compute_outputs_and_loss_test(self, input_dict, criterion, alpha, target):
        images, labels = input_dict["image"].to(self.device), input_dict["label"].to(
            self.device
        )
        train_output, train_output_domain = self.forward(images, alpha)

        loss_bce = criterion(train_output, labels)
        if target:
            labels_domain_t = (
                torch.ones(input_dict["image"].shape[0]).long().to(self.device)
            )
        else:
            labels_domain_t = (
                torch.zeros(input_dict["image"].shape[0]).long().to(self.device)
            )
        loss_domain = criterion(train_output_domain, labels_domain_t)

        return train_output, {"loss": loss_bce + alpha * loss_domain}

    # Define the learning rate scheduler function
    def lr_scheduler(self, lr, optimizer, p):
        lr = lr / (1 + 10 * p) ** 0.75
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return optimizer

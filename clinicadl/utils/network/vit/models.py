from clinicadl.utils.network.vit.vt import Transformer
from clinicadl.utils.network.sub_network import ViT
from torch import nn
import torch

MIN_NUM_PATCHES = 16


class ViTVNet(ViT):
    def __init__(
        self,
        *,
        image_size=(169, 208, 179),
        patch_size=13,
        num_classes=2,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        pool="cls",
        channels=1,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        gpu=True,
    ):
        # super().__init__(gpu=True)

        # assert all([each_dimension % patch_size ==
        # 0 for each_dimension in image_size])
        num_patches = (
            (image_size[0] - 1 // 56)
            * (image_size[1] // 52)
            * (image_size[2] - 2 // 59)
        )
        print(num_patches)
        print(MIN_NUM_PATCHES)
        patch_dim = channels * 52 * 56 * 59
        assert (
            num_patches > MIN_NUM_PATCHES
        ), f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        # num_patches = (
        #     (image_size[0] // patch_size)
        #     * (image_size[1] // patch_size)
        #     * (image_size[2] - 3 // patch_size)
        # )
        # print(num_patches)
        # print(MIN_NUM_PATCHES)
        # patch_dim = channels * patch_size**2 * 16
        # assert (
        #     num_patches > MIN_NUM_PATCHES
        # ), f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"
        # assert pool in {
        #     "cls",
        #     "mean",
        # }, "pool type must be either cls (cls token) or mean (mean pooling)"

        pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        patch_to_embedding = nn.Linear(patch_dim, dim)
        cls_token = nn.Parameter(torch.randn(1, 1, dim))
        dropout = nn.Dropout(emb_dropout)

        transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        to_latent = nn.Identity()

        mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        super().__init__(
            patch_to_embedding=patch_to_embedding,
            transformer=transformer,
            to_latent=to_latent,
            mlp_head=mlp_head,
            dropout=dropout,
            pos_embedding=pos_embedding,
            cls_token=cls_token,
            patch_size=patch_size,
            gpu=gpu,
        )

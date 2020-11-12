import numpy as np
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torch import nn

from detectron2.layers import CNNBlockBase, Conv2d, ShapeSpec, get_norm

from detectron2.layers.deform_conv import deform_conv
from detectron2.modeling import Backbone,BACKBONE_REGISTRY

cfgs = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    19: [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M"],
}

class DeformConv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        """
        Deformable convolution from :paper:`deformconv`.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(DeformConv2, self).__init__()

        assert not bias
        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)


        self.buffer_offset=nn.Sequential(
            Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(5,1),
                stride=stride_3x3,
                padding=(2*dilation,0),
                dilation=dilation,
            ),
            Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(1,5),
                stride=stride_3x3,
                padding=(0,2*dilation),
                dilation=dilation,
            ),
            nn.ReLU(inplace=True)

        )

        self.conv2_offset = Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        )

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)


    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        offset_buffer=self.buffer_offset(out)
        offset = self.conv2_offset(offset_buffer)
        out = deform_conv(
            x,
            offset,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deformable_groups,
        )
        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return x,offset

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias="+str(self.bias)
        return tmpstr


class VGG(Backbone):
    def __init__(self, stages, num_classes=None, out_features=None):
        """
        """
        super().__init__()
        self.num_classes = num_classes

        current_stride = 1
        self._out_feature_strides = {}
        self._out_feature_channels = {}

        self.stages_and_names = []
        for i, block in enumerate(stages):

            name = "res" + str(i + 1)
            stage = nn.Sequential(*block)

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            self._out_feature_strides[name] = current_stride = int(
                current_stride * 2)
            )
            self._out_feature_channels[name] = block[-1].out_channels

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.01)
                    # nn.init.constant_(m.bias, 0)
                    name = "classifier"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        outputs = {}
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
            if isinstance(x,tuple):
                x,_=x 
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.classifier(x)
            if "classifer" in self._out_features:
                outputs["classifer"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, in_channels, out_channels,pool,norm,activation):
        assert "stride" not in kwargs, "Stride of blocks in make_stage cannot be changed."
        blocks = []
        for i in range(num_blocks):
            # convert  ccp to pcc make the feature map small
            blocks.append(nn.MaxPool2d(kernel_size=2,stride=2))
            blocks.append(
                block_class[i](
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=get_norm(norm, out_channels[i]),
                    activation=activation
                )
            )

        return blocks

@BACKBONE_REGISTRY.register()
def build_points_collection_vgg_backbone(cfg, input_shape):
    # fmt: off
    depth               = cfg.MODEL.VGG.DEPTH
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    norm                = cfg.MODEL.VGG.NORM
    out_features        = cfg.MODEL.VGG.OUT_FEATURES
    in_channels         = input_shape.channels
    deform_on_per_stage = cfg.MODEL.VGG.DEFORM_ON_PER_STAGE
    # fmt: on

    stages = []
    out_stage_idx = [
        {"res1": 1, "res2": 2, "res3": 3, "res4": 4, "res5": 5}[f]
        for f in out_features
    ]
    max_stage_idx = max(out_stage_idx)
    stage_inds = [i for i, x in enumerate(cfgs[depth]) if x == "M"]
    ind = 0

    for idx, stage_idx in enumerate(range(1, max_stage_idx + 1)):

        channel_cfg=cfgs[depth][ind : stage_inds[idx]]

        if deform_on_per_stage[idx-1]:
            block_class=[Conv2d for i in range(len(channel_cfg)-1)]+[DeformConv2]
        else:
            block_class=[Conv2d for i in range(len(channel_cfg))]
        stage_kargs = {
            "block_class": block_class
            "in_channels": [in_channels]+channel_cfg[:-1],
            "out_channel": channel_cfg,
            "norm": norm,
            "pool": pool,
        }

        blocks = VGG.make_stage(**stage_kargs)
        out_channels = channel_cfg[-1]
        in_channels = out_channels
        ind = stage_inds[idx] + 1
        stages.append(blocks)
    return VGG(stages, out_features=out_features).freeze(freeze_at)
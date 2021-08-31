from torchvision import models


# just an overview if you wanna use existing models
class PtModels:
    def __init__(self, pretrained=True):
        self.pretrained = pretrained
        self.models = {
            "resnet": [
                models.resnet18(pretrained=self.pretrained),
                models.resnet34(pretrained=self.pretrained),
                models.resnet50(pretrained=self.pretrained),
                models.resnet101(pretrained=self.pretrained),
                models.resnet152(pretrained=self.pretrained),
                models.wide_resnet50_2(pretrained=self.pretrained),
                models.wide_resnet101_2(pretrained=self.pretrained),
                models.resnext50_32x4d(pretrained=self.pretrained),
                models.resnext101_32x8d(pretrained=self.pretrained),
            ],
            "alexnet": [
                models.alexnet(pretrained=self.pretrained)
            ],
            "vgg": [
                models.vgg11(pretrained=self.pretrained),
                models.vgg13(pretrained=self.pretrained),
                models.vgg16(pretrained=self.pretrained),
                models.vgg19(pretrained=self.pretrained),
                models.vgg11_bn(pretrained=self.pretrained),
                models.vgg13_bn(pretrained=self.pretrained),
                models.vgg16_bn(pretrained=self.pretrained),
                models.vgg19_bn(pretrained=self.pretrained)
            ],
            "squeezenet": [
                models.squeezenet1_0(pretrained=self.pretrained),
                models.squeezenet1_1(pretrained=self.pretrained)
            ],
            "densenet": [
                models.densenet121(pretrained=self.pretrained),
                models.densenet161(pretrained=self.pretrained),
                models.densenet169(pretrained=self.pretrained),
                models.densenet201(pretrained=self.pretrained)
            ],
            "inception": [
                models.inception_v3(pretrained=self.pretrained)
            ],
            "googlenet": [
                models.googlenet(pretrained=self.pretrained)
            ],
            "mnasnet": [
                models.mnasnet0_5(pretrained=self.pretrained),
                models.mnasnet1_0(pretrained=self.pretrained)
            ],
            "mobilenet": [
                models.mobilenet_v2(pretrained=self.pretrained)
            ]
        }
        self.pt_models = {
            "resnet": [
                models.resnet18(pretrained=self.pretrained),
                models.resnet34(pretrained=self.pretrained),
                models.resnet50(pretrained=self.pretrained),
                models.resnet101(pretrained=self.pretrained),
                models.resnet152(pretrained=self.pretrained),
                models.wide_resnet50_2(pretrained=self.pretrained),
                models.wide_resnet101_2(pretrained=self.pretrained),
                models.resnext50_32x4d(pretrained=self.pretrained),
                models.resnext101_32x8d(pretrained=self.pretrained),
            ],
            "alexnet": [
            ],
            "vgg": [],
            "squeezenet": [
                models.squeezenet1_0(pretrained=self.pretrained),
                models.squeezenet1_1(pretrained=self.pretrained)
            ],
            "densenet": [
                models.densenet121(pretrained=self.pretrained),
                models.densenet161(pretrained=self.pretrained),
                models.densenet169(pretrained=self.pretrained),
                models.densenet201(pretrained=self.pretrained)
            ],
            "inception": [
                models.inception_v3(pretrained=self.pretrained)
            ],
            "googlenet": [
                models.googlenet(pretrained=self.pretrained)
            ],
            "mnasnet": [
                models.mnasnet0_5(pretrained=self.pretrained),
                models.mnasnet1_0(pretrained=self.pretrained)
            ],
            "mobilenet": [
                models.mobilenet_v2(pretrained=self.pretrained)
            ]
        }
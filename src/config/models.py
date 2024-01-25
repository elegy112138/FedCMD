import json
from collections import OrderedDict
from typing import Dict, List, Optional, Type
from copy import deepcopy
from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .utils import PROJECT_DIR



class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.name = "DecoupledModel"
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: List[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.clone().detach())

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: torch.Tensor):
        return self.classifier(F.relu(self.base(x)))

    def get_final_features(self, x: torch.Tensor, detach=True) -> torch.Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: torch.Tensor) -> Optional[List[torch.Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    def __init__(self, dataset: str):
        super(FedAvgCNN, self).__init__()
        config = {
            "mnist": (1, 1024, 10),
            "medmnistS": (1, 1024, 11),
            "medmnistC": (1, 1024, 11),
            "medmnistA": (1, 1024, 11),
            "covid19": (3, 196736, 4),
            "fmnist": (1, 1024, 10),
            "emnist": (1, 1024, 62),
            "femnist": (1, 1, 62),
            "cifar10": (3, 1600, 10),
            "cinic10": (3, 1600, 10),
            "cifar100": (3, 1600, 100),
            "tiny_imagenet": (3, 3200, 200),
            "celeba": (3, 133824, 2),
            "svhn": (3, 1600, 10),
            "usps": (1, 800, 10),
            "domain": infer(dataset, "avgcnn"),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 512),
            )
        )
        self.classifier = nn.Linear(512, config[dataset][2])


class LeNet5(DecoupledModel):
    def __init__(self, dataset: str) -> None:
        super(LeNet5, self).__init__()
        config = {
            "mnist": (1, 256, 10),
            "medmnistS": (1, 256, 11),
            "medmnistC": (1, 256, 11),
            "medmnistA": (1, 256, 11),
            "covid19": (3, 49184, 4),
            "fmnist": (1, 256, 10),
            "emnist": (1, 256, 62),
            "femnist": (1, 256, 62),
            "cifar10": (3, 400, 10),
            "cinic10": (3, 400, 10),
            "svhn": (3, 400, 10),
            "cifar100": (3, 400, 100),
            "celeba": (3, 33456, 2),
            "usps": (1, 200, 10),
            "tiny_imagenet": (3, 2704, 200),
            "domain": infer(dataset, "lenet5"),
        }
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(config[dataset][0], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(config[dataset][1], 120),
                activation3=nn.ReLU(),
                flatten2=nn.Flatten(),
                fc2=nn.Linear(120, 84),
                activation4=nn.ReLU()
            )
        )

        self.classifier = nn.Linear(84, config[dataset][2])


class ModifiedLeNet5(DecoupledModel):
    def __init__(self, dataset: str) -> None:
        super(ModifiedLeNet5, self).__init__()
        self.name = "ModifiedLeNet5"
        config = {
            "mnist": (1, 256, 10),
            "medmnistS": (1, 256, 11),
            "medmnistC": (1, 256, 11),
            "medmnistA": (1, 256, 11),
            "covid19": (3, 49184, 4),
            "fmnist": (1, 256, 10),
            "emnist": (1, 256, 62),
            "femnist": (1, 256, 62),
            "cifar10": (3, 400, 10),
            "cinic10": (3, 400, 10),
            "svhn": (3, 400, 10),
            "cifar100": (3, 400, 100),
            "celeba": (3, 33456, 2),
            "usps": (1, 200, 10),
            "tiny_imagenet": (3, 2704, 200),
            "domain": infer(dataset, "lenet5"),
        }
        config2 = {
            "mnist": 864,
            "medmnistC": 864,
            "medmnistA": 864,
            "fmnist": 864,
            "emnist": 864,
            "cifar10": 1176,
            "cinic10": 1176,
            "svhn": 1176,
            "cifar100": 1176,
            "tiny_imagenet": 5400,
        }

        self.config = deepcopy(config[dataset])
        # # 分解网络结构
        self.conv1 = nn.Sequential(
            nn.Conv2d(config[dataset][0], 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config[dataset][1], 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU()
        )

        # self.fc3 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(84, 84),
        #     nn.ReLU()
        # )
        #
        # self.fc4 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(84, 84),
        #     nn.ReLU()
        # )

        self.classifier = nn.Linear(84, config[dataset][2])

    def check_avaliability(self):
        pass

    def forward(self, x):
        # 保存每一层的输出
        outputs = []
        # outputs.append(x)

        x1 = self.conv1(x)
        # outputs.append(x1)

        x2 = self.conv2(x1)
        outputs.append(x2)

        x3 = self.fc1(x2)
        outputs.append(x3)

        x4 = self.fc2(x3)
        outputs.append(x4)

        # x5 = self.fc3(x4)
        # outputs.append(x5)
        # # #
        # x6 = self.fc4(x5)
        # outputs.append(x6)

        logits = self.classifier(x4)
        outputs.append(logits.float())

        # 将输出写入 CSV
        return outputs
        # return logits


    def normalize_tensor(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = ((tensor - min_val) / (max_val - min_val))
        return normalized_tensor

    def determine_max_contribution_layer(self, outputs, x, y):
        kl_divergences_y, delta_kl_div_y, kl_divergences_x, delta_kl_div_x = [], [], [], []
        for i, output in zip(range(len(outputs[0:])), outputs[0:]):
            kl_div_x = self.compute_kl_divergence(output, x)
            kl_div_y = self.compute_kl_divergence(output, y)
            kl_divergences_x.append(kl_div_x)
            kl_divergences_y.append(kl_div_y)

        delta_kl_div_x = [abs(kl_divergences_x[i] - kl_divergences_x[i - 1]) for i in range(len(kl_divergences_x)) if
                          i > 0]
        delta_kl_div_y = [abs(kl_divergences_y[i] - kl_divergences_y[i - 1]) for i in range(len(kl_divergences_y)) if
                          i > 0]

        # 选择delta_KL散度最小的层
        delta_kl_div = np.array(
            [abs(y - x) for x, y in zip(delta_kl_div_x, delta_kl_div_y)])
        delta_kl_div = list(delta_kl_div)
        if delta_kl_div[1] == min(delta_kl_div):  # 只要fc2最小，即便有多个最小值，也取fc2（先验知识）
            min_kl_index = 1
        else:
            min_kl_index = delta_kl_div.index(min(delta_kl_div))
        layers = ["fc1", "fc2", "classifier"]
        return layers[min_kl_index]

    def compute_kl_divergence(self, layer_output, y):
        # 归一化y
        y = self.normalize_tensor(y.float())
        layer_output = self.normalize_tensor(layer_output.float())
        layer_mean, layer_std = layer_output.mean(), layer_output.std()
        y_mean, y_std = y.mean(), y.std()
        if torch.isnan(layer_mean).any() \
                or torch.isnan(layer_std).any() \
                or torch.isnan(y_mean).any() \
                or torch.isnan(y_std).any():
            return 0

        layer_dist = Normal(layer_mean, layer_std)
        y_dist = Normal(y_mean, y_std)

        Wasserstein_distance = torch.norm(layer_dist.mean - y_dist.mean, p=2)
        return Wasserstein_distance.item()


class TwoNN(DecoupledModel):
    def __init__(self, dataset):
        super(TwoNN, self).__init__()
        config = {
            "mnist": (784, 10),
            "medmnistS": (784, 11),
            "medmnistC": (784, 11),
            "medmnistA": (784, 11),
            "fmnist": (784, 10),
            "emnist": (784, 62),
            "femnist": (784, 62),
            "cifar10": (3072, 10),
            "cinic10": (3072, 10),
            "svhn": (3072, 10),
            "cifar100": (3072, 100),
            "usps": (1536, 10),
            "synthetic": (60, 10),  # default dimension and classes
        }

        self.base = nn.Linear(config[dataset][0], 200)
        self.classifier = nn.Linear(200, config[dataset][1])
        self.activation = nn.ReLU()

    def need_all_features(self):
        return

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.base(x))
        x = self.classifier(x)
        return x

    def get_final_features(self, x, detach=True):
        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        x = torch.flatten(x, start_dim=1)
        x = self.base(x)
        return func(x)

    def get_all_features(self, x):
        raise RuntimeError("2NN has 0 Conv layer, so is unable to get all features.")


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, 1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, dataset):
        super(MobileNetV2, self).__init__()
        self.outputs = []  # 用于保存每一层的输出
        self.name = 'MobileNetV2'
        # 数据集配置信息
        config = {
            "mnist": (1, 28, 10),
            "medmnistS": (1, 256, 11),
            "medmnistC": (1, 256, 11),
            "medmnistA": (1, 256, 11),
            "covid19": (3, 49184, 4),
            "fmnist": (1, 28, 10),
            "emnist": (1, 28, 62),
            "femnist": (1, 28, 62),
            "cifar10": (3, 32, 10),
            "cinic10": (3, 32, 10),
            "svhn": (3, 32, 10),
            "cifar100": (3, 32, 100),
            "celeba": (3, 218, 2),
            "usps": (1, 16, 10),
            "tiny_imagenet": (3, 64, 200),
            "domain": infer(dataset, "lenet5"),  # 你可能需要定义一个infer函数来获取相应的配置
        }

        input_channels, input_size, num_classes = config.get(dataset, (3, 224, 1000))

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 320, 1, 6),
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )

        # Classifier
        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 640),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 320),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        self.outputs = []  # 清空之前的输出
        x1 = self.features(x)
        self.outputs.append(x1)
        x2 = self.fc1(x1)
        self.outputs.append(x2)
        x3 = self.fc2(x2)
        self.outputs.append(x3)
        x4 = self.classifier(x3)
        self.outputs.append(x4)
        return self.outputs

    def check_avaliability(self):
        pass


class ResNet18(DecoupledModel):
    def __init__(self, dataset):
        super(ResNet18, self).__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "res18"),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(self.base.fc.in_features, config[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().forward(x)

    def get_all_features(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


class AlexNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        config = {
            "covid19": 4,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "alex"),
        }
        if dataset not in config.keys():
            raise NotImplementedError(f"AlexNet does not support dataset {dataset}")

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.alexnet(
            weights=models.AlexNet_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[-1].in_features, config[dataset]
        )
        self.base.classifier[-1] = nn.Identity()


class SqueezeNet(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        config = {
            "mnist": 10,
            "medmnistS": 11,
            "medmnistC": 11,
            "medmnistA": 11,
            "fmnist": 10,
            "svhn": 10,
            "emnist": 62,
            "femnist": 62,
            "cifar10": 10,
            "cinic10": 10,
            "cifar100": 100,
            "covid19": 4,
            "usps": 10,
            "celeba": 2,
            "tiny_imagenet": 200,
            "domain": infer(dataset, "sqz"),
        }

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        sqz = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        )
        self.base = sqz.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(sqz.classifier[1].in_channels, config[dataset], kernel_size=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return self.classifier(self.base(x))

    def get_all_features(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)


# Some dirty codes for adapting DomainNet
def infer(dataset, model_type):
    if dataset == "domain":
        with open(PROJECT_DIR / "data" / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
        class_num = metadata["class_num"]
        img_size = metadata["image_size"]
        coef = {"avgcnn": 50, "lenet5": 42.25}
        if model_type in ["alex", "res18", "sqz", "mobile"]:
            return class_num
        return (3, int(coef[model_type] * img_size), class_num)


MODEL_DICT: Dict[str, Type[DecoupledModel]] = {
    "lenet5": LeNet5,
    "ModifiedLeNet5": ModifiedLeNet5,
    "avgcnn": FedAvgCNN,
    "2nn": TwoNN,
    "mobile": MobileNetV2,
    "res18": ResNet18,
    "alex": AlexNet,
    "sqz": SqueezeNet,
}

from torchvision import models


def efficientnet_b2(num_classes=1000):
    return models.efficientnet_b2(num_classes=num_classes)


def mobilenet(num_classes=1000):
    return models.mobilenet_v3_small()

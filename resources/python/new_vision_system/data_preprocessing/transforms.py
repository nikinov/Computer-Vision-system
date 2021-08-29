from torchvision import transforms

class MyTransforms():
    def __init__(self):
        pass
    def get_train_transforms(self, input_size):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
                transforms.RandomResizedCrop(size=(input_size + 20, input_size + 20), scale=(0.9, 1.1), ratio=(0.4, 0.6)),
                transforms.Resize((input_size, input_size)),
                transforms.Normalize((0.5,), (0.5,))
            ])
    def get_gray_train_transforms(self, input_size):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
                transforms.RandomResizedCrop(size=(195, 100), scale=(0.9, 1.1), ratio=(0.4, 0.6)),
                transforms.Grayscale(),
                transforms.Resize((input_size, input_size)),
                transforms.Normalize((0.5,), (0.5,))
            ])
    def get_EfficientNet_train_transfroms(self, advprop=False):
        if advprop:  # for models using advprop pretrained weights
            return transforms.Lambda(lambda img: img * 2.0 - 1.0)
        else:
            return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


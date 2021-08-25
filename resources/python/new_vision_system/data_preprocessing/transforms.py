from torchvision import transforms

class LinearTransform():
    def __init__(self):
        pass
    def get_train_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
                transforms.RandomResizedCrop(size=(195, 100), scale=(0.9, 1.1), ratio=(0.4, 0.6)),
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.Normalize((0.5,), (0.5,))
            ])
    def get_valid_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.Normalize((0.5,), (0.5,))
            ])

class ResnetTransforms():
    def __init__(self):
        pass
    def get_train_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=.1, contrast=.1, hue=.1),
                transforms.RandomResizedCrop(size=(234, 234), scale=(0.9, 1.1), ratio=(0.4, 0.6)),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5,), (0.5,))
            ])
    def get_valid_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5,), (0.5,))
            ])

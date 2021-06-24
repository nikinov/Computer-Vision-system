import torch
import cv2 as cv
import numpy as np
import os
from torchvision import transforms

path = "otherImages/_Bad_Void"

image = cv.imread(os.path.join(path, "01055538070_testfield_1028_2.bmp"))
image2 = cv.imread(os.path.join(path, "01055542031_testfield_1028_6.bmp"))

b,g,r = cv.split(image)
b2,g2,r2 = cv.split(image2)

col = np.vstack([r, g, b])
col2 = np.vstack([r2, g2, b2])

full = np.hstack([col, col2])

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
}


model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
model.eval()

train_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(full)























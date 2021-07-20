from torch import nn, optim
import torch
import glob
import cv2
from torchvision import transforms
import numpy as np

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

directory = "C:/!SAMPLES!/1716-5082/Assets5082-samples-for-training/Assets5082"
class_num = 0
classes = {}
k=2
attemts=10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

input_size = 784
hidden_sizes = [128, 64]
output_size = 11
val_data = []
train_data = []

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

for i, entry in enumerate(glob.iglob(directory + '**/**')):
    classes[entry.split(r"\ ".replace(" ", ""))[-1]] = i

class_check = ""
for i, entry in enumerate(glob.iglob(directory + '**/**', recursive=True)):
    if entry.endswith(".bmp"):
        img = cv2.imread(entry, 0)
        img = cv2.resize(img, (28, 28))
        img = np.float32(img.reshape(1, input_size))
        ret, label, center = cv2.kmeans(img, k, None, criteria, attemts, cv2.KMEANS_PP_CENTERS)
        label = label*(255/(k-1))
        label = transform(label)
        splitter = r"\ ".replace(" ","")
        if entry.split(splitter)[-2] != class_check:
            class_check = entry.split(splitter)[-2]
            val_data.append([label, entry.split(splitter)[-2]])
        else:
            train_data.append([label, classes[entry.split(splitter)[-2]]])


print(len(val_data))
print(len(train_data))
model.to("cuda")

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in train_data:

        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        images = images.type(torch.FloatTensor)
        images = images.to("cuda")
        labels = torch.tensor(labels)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels.to("cuda"))

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(val_data)))

    correct_count, all_count = 0, 0
    for images, labels in val_data:
        with torch.no_grad():
            logps = model(images.to("cuda"))

        ps = torch.exp(logps).to("cuda")
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1
    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))



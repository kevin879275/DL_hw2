import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

dataFile = 'data.npy'
labelFile = 'label.npy'
Epoch = 50
BATCH_SIZE = 2


class WaferMapDataset(Dataset):
    def __init__(self, transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.data = np.load(dataFile)
        self.data = self.data.astype('float32')
        self.label = np.load(labelFile)
        self.transform = transform
        #self.data = transform(self.data)
        #self.data = self.data.reshape(len(self.data), 3, 26, 26)
        #self.data = np.transpose(self.data, (0, 3, 1, 2))

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.data)


transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = WaferMapDataset(transform=transform)

train_loader = DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 1)
#         self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
#         self.conv2 = nn.Conv2d(32, 32, 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x, index1 = self.maxpool(x)
#         x,


# class Decoder(nn.Module):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         pass


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 2)
        self.deconv2 = nn.ConvTranspose2d(32, 3, 2)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_conv = self.relu(self.conv1(x))
        x, index1 = self.maxpool(x_conv)
        x1 = self.relu(self.conv2(x))
        latent_code, index2 = self.maxpool(x1)
        #y = self.upsample(latent_code)
        y = self.unpool(latent_code, index2, output_size=x1.size())
        y = self.relu(self.deconv1(y))
        y = self.unpool(y, index1, output_size=x_conv.size())
        #y = self.upsample(y)
        out = self.relu(self.deconv2(y))
        return out


model = AutoEncoder()
print(model)
loss = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

print("----------- training start -----------")
for epoch in range(Epoch):
    total_loss = 0
    loss_val = 0
    for imgs, label in train_loader:
        optimizer.zero_grad()
        out = model(imgs)
        out_np = out.detach().numpy()
        out_np = out_np[0].reshape(26, 26, 3)
        loss_val = loss(out, imgs)
        total_loss += loss_val
        loss_val.backward()
        optimizer.step()
    print("epoch  = ", epoch + 1, "loss  = ",
          total_loss.item() / len(train_loader))


print("------------ test start ---------------")

for imgs, label in train_loader:
    out = model(imgs)
    out_np = out.detach().numpy()
    out_np = out_np[0].transpose((2, 1, 0))
    plt.imshow(np.argmax(out_np, axis=2))
    # plt.imshow(out_np)
    plt.show()
    break

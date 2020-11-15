import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

dataFile = 'data.npy'
labelFile = 'label.npy'
Epoch = 30
BATCH_SIZE = 4


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


'''
Initial data transform
'''
transform = transforms.Compose([
    transforms.ToTensor(),
])

'''
Create Dataset
'''
train_set = WaferMapDataset(transform=transform)

'''
Create DataLoader with batch_size and shuffle
'''
train_loader = DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

'''
Autoencoder network architecture
'''


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 2, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 16, 2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 32, 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 3, 2, padding=1)
        self.unpool = nn.MaxUnpool2d(2, 2)
        #self.upsample = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_conv = self.sigmoid(self.conv1(x))
        x, index1 = self.maxpool(x_conv)
        #Norm1 = nn.BatchNorm2d(x.shape[1])
        #x = Norm1(x)
        x1 = self.sigmoid(self.conv2(x))
        latent_code, index2 = self.maxpool(x1)
        #Norm2 = nn.BatchNorm2d(latent_code.shape[1])
        #latent_code = Norm2(latent_code)
        y = self.unpool(latent_code, index2, output_size=x1.size())
        y = self.sigmoid(self.deconv1(y))
        y = self.unpool(y, index1, output_size=x_conv.size())
        out = self.sigmoid(self.deconv2(y))
        return out


model = AutoEncoder()
print(model)
'''
using MSE loss function
'''
loss = nn.MSELoss()
'''
Using AdamW optimizer with learning rate = 0.01
'''
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


test_set = WaferMapDataset(transform=transform)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

classes = []
test_data_res = []
test_label_res = []
num_of_gen = 5

for imgs, label in test_loader:
    if int(label[0]) not in classes:
        classes.append(int(label[0]))
        plt.figure()
        plt.subplot(1, 6, 1)
        imgs_np = imgs.detach().numpy()
        imgs_show = imgs_np[0].transpose((1, 2, 0))
        plt.imshow(np.argmax(imgs_show, axis=2))
        for i in range(num_of_gen):
            imgs_noise = imgs + torch.normal(mean=0, std=0.1, size=imgs.size())
            out = model(imgs_noise)
            out_np = out.detach().numpy()
            out_np = out_np[0].transpose((1, 2, 0))
            test_data_res.append(out_np)
            test_label_res.append(label)
            plt.subplot(1, 6, i + 2)
            plt.imshow(np.argmax(out_np, axis=2))
        plt.show()
    else:
        for i in range(num_of_gen):
            imgs_noise = imgs + torch.normal(mean=0, std=0.1, size=imgs.size())
            out = model(imgs_noise)
            out_np = out.detach().numpy()
            out_np = out_np[0].transpose((1, 2, 0))
            test_data_res.append(out_np)
            test_label_res.append(label)


'''
save result into .npy file
'''
np.save('test_data_result.npy', test_data_res)
np.save('test_label_result.npy', test_label_res)

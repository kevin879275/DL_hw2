import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataFile = 'data.npy'
labelFile = 'label.npy'
Epoch = 20
BATCH_SIZE = 32

'''
Create WaferMapDataset
'''


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
        self.conv1 = nn.Conv2d(3, 64, 2, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2 = nn.Conv2d(64, 16, 2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(16, 64, 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 3, 2, padding=1)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_conv = self.relu(self.conv1(x))
        x, index1 = self.maxpool(x_conv)
        x1 = self.relu(self.conv2(x))
        latent_code, index2 = self.maxpool(x1)
        y = self.unpool(latent_code, index2, output_size=x1.size())
        y = self.relu(self.deconv1(y))
        y = self.unpool(y, index1, output_size=x_conv.size())
        out = self.relu(self.deconv2(y))
        return out


model = AutoEncoder().to(device)
# print(model)
summary(model, (3, 26, 26))
'''
using MSE loss function
'''
loss = nn.MSELoss()
'''
Using AdamW optimizer with learning rate = 0.01
'''
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

print("----------- training start -----------")
loss_val_list = []
for epoch in range(Epoch):
    total_loss = 0
    loss_val = 0
    for imgs, label in train_loader:
        imgs = imgs.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        # out_np = out.detach().numpy()
        # out_np = out_np[0].reshape(26, 26, 3)
        loss_val = loss(out, imgs)
        total_loss += loss_val
        loss_val.backward()
        optimizer.step()
    loss_val_list.append(total_loss.item() / len(train_loader))
    print("epoch  = ", epoch + 1, "loss  = ",
          total_loss.item() / len(train_loader))
# draw loss value per epoch
plt.plot(range(1, Epoch + 1), loss_val_list)
plt.title("loss value")
plt.ylabel("loss value")
plt.xlabel("epoch")
plt.show()

print("------------ test start ---------------")


test_set = WaferMapDataset(transform=transform)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

classes = []
test_data_res = []
test_label_res = []
# generate five image per data
num_of_gen = 5

# class label
label_classes = {0: 'Center', 1: 'Dount', 2: 'Edge-Loc', 3: 'Edge-Ring',
                 4: 'Loc', 5: 'Near-full', 6: 'Random', 7: 'Scratch', 8: 'None'}

for imgs, label in test_loader:
    # if this label image not shown
    imgs = imgs.to(device)
    label = label.to(device)
    if int(label[0]) not in classes:
        # save label if shown
        classes.append(int(label[0]))
        # show original image
        plt.figure()
        plt.subplot(1, 6, 1)
        plt.title(label_classes[classes[-1]])
        imgs_np = imgs.cpu().detach().numpy()
        imgs_show = imgs_np[0].transpose((1, 2, 0))
        # show image with (max value of dim channel)
        plt.imshow(np.argmax(imgs_show, axis=2))
        for i in range(num_of_gen):
            # add noise
            imgs_noise = imgs + \
                torch.normal(mean=0, std=0.1, size=imgs.size()).to(device)
            out = model(imgs_noise)
            out_np = out.cpu().detach().numpy()
            # (C,H,W)->(H,W,C) image format
            out_np = out_np[0].transpose((1, 2, 0))
            test_data_res.append(out_np)
            test_label_res.append(label)
            # show generate image
            plt.subplot(1, 6, i + 2)
            plt.imshow(np.argmax(out_np, axis=2))
        plt.show()
    else:  # other result without show image
        for i in range(num_of_gen):
            imgs_noise = imgs + \
                torch.normal(mean=0, std=0.1, size=imgs.size()).to(device)
            out = model(imgs_noise)
            out_np = out.cpu().detach().numpy()
            out_np = out_np[0].transpose((1, 2, 0))
            test_data_res.append(out_np)
            test_label_res.append(label)


'''
save result into .npy file
'''
np.save('test_data_result.npy', test_data_res)
np.save('test_label_result.npy', test_label_res)

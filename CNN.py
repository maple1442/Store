import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from scipy import ndimage
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt

# x = np.load('Wafermap.npy')
# y = np.load('Label.npy')
# print('x shape : {}, y shape : {}'.format(x.shape, y.shape))

# # Filiter None
# faulty_case = np.unique(y)
# idx = ~(y.squeeze(1)==faulty_case[-1])
# print(idx)
# x = x[idx]
# y = y[idx]
# np.save('CleanWafermap.npy', x)
# np.save('CleanLabel.npy',y)

x = np.floor(np.load('CleanWafermap.npy') * 127.5).astype(np.uint8)
y = np.load('CleanLabel.npy')
faulty_case = np.unique(y)
print('x shape : {}, y shape : {}'.format(x.shape, y.shape))
print('Faulty case list : {}'.format(faulty_case))
for i ,f in enumerate(faulty_case) :
    print('{} : {}'.format(f, len(y[y==f])))
faulty_case = faulty_case[[0,1,2,3,4,6,7,5]]
print(faulty_case)
le = preprocessing.LabelEncoder()
le.fit(faulty_case)
Label = le.transform(y)
print(np.unique(Label))

class WaferDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

class Naigong_CNN(nn.Module):
    def __init__(self):
        super(Naigong_CNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3, padding=0),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3, padding=0),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3, padding=0),
        )
        self.linear = nn.Sequential(
            nn.Linear(8 * 8 * 128, 4096),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(1024, 8),
        )
    def forward(self, x):
        x = self.CNN(x)
        x = torch.flatten(x, start_dim=1)
        out = self.linear(x)
        return out

class Median:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        sample = ndimage.median_filter(sample, size = self.kernel_size)
        return sample

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    Median(kernel_size = 9),
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    # # transforms.RandomHorizontalFlip(),
    # # transforms.RandomRotation(15),
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    Median(kernel_size = 9),
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),                       
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
])

batch_size = 128

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
Total = len(train_x) + len(val_x) + len(test_x)

print("Training : {} [{:.2f}]".format(len(train_x), len(train_x)/Total))
print("Validation : {} [{:.2f}]".format(len(val_x), len(val_x)/Total))
print("Testing : {} [{:.2f}]".format(len(test_x), len(test_x)/Total))
hist_train = []
hist_val = []
hist_test = []
for i ,f in enumerate(faulty_case) :
    print('{} : {}, {}, {}'.format(f, len(train_y[train_y==f]), len(val_y[val_y==f]), len(test_y[test_y==f])))
    hist_train.append(len(train_y[train_y==f]))
    hist_val.append(len(val_y[val_y==f]))
    hist_test.append(len(test_y[test_y==f]))

x=np.arange(len(faulty_case))
width=0.25
ax = plt.figure(figsize=(10, 10))
plt.bar(x, hist_train, tick_label=faulty_case, color = "blue", label="Train", width=0.25)
x1=[p + width for p in x]
plt.bar(x1, hist_val, tick_label=faulty_case, color = "orange", label="Val", width=0.25)
x2=[p + width for p in x1]
plt.bar(x2, hist_test, tick_label=faulty_case, color = "gray", label="Test", width=0.25)
plt.xticks([p + width/3 for p in x], x)

for a,b in zip(x,hist_train):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=8)
for a,b in zip(x1,hist_val):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=8)
for a,b in zip(x2,hist_test):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=8)

plt.xticks(x, faulty_case)
plt.legend()
plt.savefig("Hist.png")


train_y = le.transform(train_y)
val_y = le.transform(val_y)
test_y = le.transform(test_y)

train_set = WaferDataset(train_x, train_y, train_transform)
val_set = WaferDataset(val_x, val_y, test_transform)
test_set = WaferDataset(test_x, test_y, test_transform)
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 4)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 4)

model = Naigong_CNN().cuda()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
num_epoch = 100

from torchsummary import summary
summary(model, (3, 224, 224))

best_acc = 0
Train_Loss_list = []
Valid_Loss_list = []

# def preprocessing(input_wfr):
    # input_wfr = input_wfr - 1
    # t_data = ndimage.median_filter(t_data, size =9)

import pdb
from torchvision import utils as utils
# utils.save_image(data[0], "batch.png", normalize = True)
# for epoch in range(num_epoch):
#     epoch_start_time = time.time()
#     train_acc = 0.0
#     train_loss = 0.0
#     val_acc = 0.0
#     val_loss = 0.0
#     model.train()
#     for i, data in enumerate(train_loader):
#         train_pred = model(torch.FloatTensor(data[0]).cuda())
#         batch_loss = loss(train_pred, data[1].cuda())
#         optimizer.zero_grad()
#         batch_loss.backward() 
#         optimizer.step()
#         train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
#         train_loss += batch_loss.item()
    
#     model.eval()
#     with torch.no_grad():
#         for i, data in enumerate(val_loader):
#             val_pred = model(torch.FloatTensor(data[0]).cuda())
#             batch_loss = loss(val_pred, data[1].cuda())
#             val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
#             val_loss += batch_loss.item()
#         if val_acc > best_acc:
#             torch.save(model.state_dict(), './save/CNN_Best.pth')
#             best_acc = val_acc

#         #將結果 print 出來
#         print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
#             (epoch + 1, num_epoch, time.time()-epoch_start_time, \
#              train_acc/train_set.__len__(), train_loss/len(train_loader), val_acc/val_set.__len__(), val_loss/len(val_loader)))
#         Train_Loss_list.append(train_loss/train_set.__len__())
#         Valid_Loss_list.append(val_loss/val_set.__len__())

# torch.save(model.state_dict(), './save/CNN_{}.pth'.format(epoch))

model.load_state_dict(torch.load('./save/CNN_Best.pth'))
model.eval()
test_acc = 0
with torch.no_grad():
    for i, data in enumerate(val_loader):
        test_pred = model(torch.FloatTensor(data[0]).cuda())
        batch_loss = loss(test_pred, data[1].cuda())
        test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
    print(test_acc)
    print(val_set.__len__())
    #將結果 print 出來
    print('Test Acc: %3.6f' % (test_acc/val_set.__len__()))
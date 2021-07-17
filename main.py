from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
from AI.DataSet import Data
from AI.Net import Net

img_folder = 'path_to_photos'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

classes = ('White', 'Nigger', 'Asian')

dataS = Data(img_folder, transform=transform)

train_loader = DataLoader(dataS, batch_size=batch_size, shuffle=False,
                          num_workers=4, drop_last=True)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i% 2000 == 1999:
            print('[%d, %5d} loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('FINISH!')
save = 'path_to_folder'

torch.save(net.state_dict(), save)

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

from model import Odevity
from dataset import MyData

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = torch.load("./1.pth")

datas = MyData("./valid.csv")

datal = DataLoader(datas,batch_size=1)

model.eval()

total = 0
correct = 0

for data in datal:
    number, label = data
    number = number.to(device)
    label = label.to(device)
    output = model(number)
    total += 1
    print(output[0], label[0])
    if output[0] >= 0.5 and label[0] == 1 or output[0] < 0.5 and label[0] == 0:
        correct += 1

print(correct / total)
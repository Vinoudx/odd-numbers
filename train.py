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

model = Odevity().to(device)

datas = MyData("./data.csv")

datal = DataLoader(datas, batch_size=10, shuffle=True, drop_last=False)

optimizer = Adam(model.parameters(), lr=0.01)

loss_fun = MSELoss()
epoch = 10

for i in range(epoch):
    print("epoch:{}".format(i))
    for data in datal:
        number, label = data
        number = number.to(device)
        label = label.to(device)

        output = model(number)
        loss = loss_fun(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss:{}".format(loss))

torch.save(model, "./1.pth")

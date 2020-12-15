from glob import glob
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

#고정 파라미터 x,y 만드는 함수
def get_data():
    train_X = np.asarray(
        [3.3, 4.4, 5.5, 6.72, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray(
        [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    dtype = torch.FloatTensor

    X = Variable(torch.from_numpy(train_X).type(dtype), requires_grad=False).view(17, 1)

    y = Variable(torch.from_numpy(train_Y).type(dtype), requires_grad=False)

    print("X :", X)
    print("y :", y)
    return X, y

#학습 파라미터 w,b를 만드는 함수
def get_weights():
    w = Variable(torch.randn(1), requires_grad=True)
    b = Variable(torch.randn(1), requires_grad=True)
    print("w : ", w)
    print("b : ", b)

    return w, b


def simple_network(x, w, b):
    y_pred = torch.matmul(x, w) + b
    return y_pred


def loss_fn(y, y_pred, w, b):
    loss = (y_pred - y).pow(2).sum()
    for param in [w, b]:
        if not param.grad is None: param.grad.data.zero_()

    # if not loss.data is None:
    #     loss.data.zero_()
    loss.backward()

    return loss.data


def optimize(learning_rate, w, b):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


class DogsAndCatsDataset(Dataset):
    def __init__(self, root_dir, size=(224, 224)):
        self.files = glob(root_dir)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split('/')[-2]
        return img, label

# dataloader = DataLoader(dogsdset, batch_size=32, num_workers=2)
# for imgs, labels in dataloader:
#     pass

if __name__ == "__main__":
    x, y = get_data()
    w, b = get_weights()

    y_pred = simple_network(x, w, b)

    loss_data = loss_fn(y, y_pred, w, b)
    print("y_pred : ", y_pred)
    print("loss_data", loss_data)

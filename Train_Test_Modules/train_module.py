import torch
from torch import nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

from model.iResnet import iresnet50
from model.fmobilenet import MobileFaceNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from metrics import ArcMarginProduct, ArcFace
import numpy as np

torch.autograd.set_detect_anomaly(True)  # 自动检测nan


# 定义FocalLoss
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


# 定义训练集
class TrainDataSet(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        # 将category映射到label
        self.category_reflect = {}
        for label, category in enumerate(os.listdir(path)):
            self.category_reflect[category] = label

        self.dataset = []
        for category in os.listdir(path):
            category_path = os.path.join(path, category) + '/'
            category_label = self.category_reflect[category]
            for face in os.listdir(category_path):
                face_path = os.path.join(category_path, face)
                if os.path.splitext(face_path)[-1] == '.jpg':
                    img = Image.open(face_path)
                    face_tensor = transform(img)
                    self.dataset.append((face_tensor, category_label))
        print('all train data has been loaded')

    def __getitem__(self, index):
        face_tensor, label = self.dataset[index]
        return face_tensor, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    '''定义transform'''
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(112),
        transforms.ToTensor(),
        # transforms.Lambda()
        transforms.Normalize(norm_mean, norm_std), ])

    '''定义训练集'''
    train_path = r"data\train_cut_result"
    batch_size = 8  # batch_size初始化
    train_dataset = TrainDataSet(train_path, transform)
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)

    '''定义训练方式'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU applied ')
    else:
        print('CPU applied')
        device = torch.device('cpu')

    # device = torch.device('cpu')

    '''定义网络'''
    net = iresnet50(dropout=0.0, fp16=True, num_features=512)  # 网络实例化并传入cuda
    # num_features = net.fc.in_features
    # net.fc = nn.Linear(num_features, 512)

    net = net.to(device)
    params_load_path = r"params\face_discern.pth"  # 参数的保存路径
    params_save_path = r"params\face_discern_res.pth"
    if os.path.exists(params_load_path):
        net.load_state_dict(torch.load(params_load_path))
        print('Params loaded')
    else:
        print("NO Params")

    '''参数初始化'''
    learning_rate = 1e-1  # 学习率初始化
    lr_step = 10  # 每10个epoch学习下降一次
    lr_decay = 0.95  # lr = lr*lr_decay
    weight_decay_rate = 5e-4
    epoch_num = 100

    '''优化器、学习率、loss函数'''
    # arc_face = ArcMarginProduct(in_features=512, out_features=152).to(device)  # Arcface_metrics
    arc_face = ArcFace().to(device)  # Arcface_loss
    optimizer = optim.SGD(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay_rate,
        momentum=0.9)  # 优化器
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_decay)  # 学习率控制
    loss_func = FocalLoss().to(device)  # loss函数

    '''训练开始'''
    iter_num = 0
    for epoch in range(epoch_num):

        count = 0
        loss_sum = 0
        net.train()
        for features, labels in train_iter:
            features, labels = features.to(device), labels.to(device)
            # print(features.shape)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()  # 释放GPU缓存
            logits = net(features)
            assert torch.isnan(logits).sum() == 0, print(logits)
            logits = F.normalize(logits)
            assert torch.isnan(logits).sum() == 0, print(logits)
            logits = arc_face.forward(logits, labels)
            assert torch.isnan(logits).sum() == 0, print(logits)
            loss = loss_func(logits, labels).sum()
            assert torch.isnan(loss).sum() == 0, print(loss)

            optimizer.zero_grad()
            # clip_grad_norm_(net.parameters(), max_norm=5, norm_type=2)  # 梯度截断
            loss.backward()
            optimizer.step()
            loss_sum += loss.data.item()
            count += 1
            print('epoch: %2d' % epoch, ' count: %2d' % count, ' loss: %6.3lf' % loss.data.item())
            iter_num += 1
        scheduler.step()

        '''保存参数'''
        if epoch % 5 == 0 or epoch == epoch_num - 1:
            torch.save(net.state_dict(), params_save_path)
            print("第 {} 轮参数保存成功！".format(epoch))
            print(str(epoch) + ' Train_ave_Loss: %.3lf' % (loss_sum / count))
        else:
            print(str(epoch) + ' Train_Loss : %.3lf' % (loss_sum / count))
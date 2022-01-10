import os.path as osp

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import torchvision.transforms as T

from model.iResnet import iresnet50
from model.fmobilenet import MobileFaceNet


def unique_image(pair_list) -> set:
    """Return unique image path in pair_list.txt"""
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    unique = set()
    for pair in pairs:
        id1, id2, _ = pair.split()
        unique.add(id1)
        unique.add(id2)
    return unique


def group_image(images: set, batch) -> list:
    """Group image paths by batch size"""
    images = list(images)
    size = len(images)
    res = []
    for i in range(0, size, batch):
        end = min(batch + i, size)
        res.append(images[i: end])
    return res


def _preprocess(images: list, transform) -> torch.Tensor:
    """预处理函数"""
    res = []
    img_tensors = torch.empty(test_batch_size, 3, 112, 112)
    rec = 0
    for img in images:
        im = Image.open(img)
        im = transform(im)
        res.append(im)
        img_tensors[rec, :, :, :] = im
        rec += 1
    # data = torch.cat(res, dim=0)  # shape: (batch, 128, 128)
    # print(img_tensors.shape)
    # data = data[:, None, :, :]  # shape: (batch, 1, 128, 128)
    return img_tensors


def featurize(images: list, transform, net, device) -> dict:
    """特征输入模型"""
    data = _preprocess(images, transform)
    data = data.to(device)
    net = net.to(device)
    with torch.no_grad():
        features = net(data)
    res = {img: feature for (img, feature) in zip(images, features)}
    return res


def cosin_metric(x1, x2):
    """计算两个特征间的余弦距离"""
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def threshold_search(y_score, y_true):
    """人脸划分"""
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


def compute_accuracy(feature_dict, pair_list, test_root):
    """计算准确率"""
    with open(pair_list, 'r') as f:
        pairs = f.readlines()

    similarities = []
    labels = []
    for pair in pairs:
        img1, img2, label = pair.split()
        img1 = osp.join(test_root, img1)
        img2 = osp.join(test_root, img2)
        feature1 = feature_dict[img1].cpu().numpy()
        feature2 = feature_dict[img2].cpu().numpy()
        label = int(label)

        similarity = cosin_metric(feature1, feature2)
        similarities.append(similarity)
        labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold


if __name__ == '__main__':
    embedding_size = 512
    drop_ratio = 0.5

    '''预处理'''
    input_shape = [3, 112, 112]
    test_transform = T.Compose([
        # T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    '''测试数据'''
    test_root = "data/lfw-align-128"
    test_list = "data/lfw_test_pair.txt"
    test_batch_size = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(str(device) + ' loaded')

    '''加载模型'''
    # model = MobileFaceNet(512) # 使用mobilefacenet
    # model = nn.DataParallel(model) # 注：使用mobilenet需要使用灰度图，更改预处理函数
    model = iresnet50(dropout=0.0, fp16=True, num_features=512)  # 使用iresnet50
    test_model_path = "params/ir50_glt.pth"
    model.load_state_dict(torch.load(test_model_path))
    model.to(device)
    print(test_model_path + ' loaded')
    model.eval()

    '''加载数据'''
    images = unique_image(test_list)
    images = [osp.join(test_root, img) for img in images]
    groups = group_image(images, test_batch_size)
    print('data loaded')

    '''开始测试'''
    feature_dict = dict()
    print('test started...')
    for group in groups:
        d = featurize(group, test_transform, model, device)
        feature_dict.update(d)
    accuracy, threshold = compute_accuracy(feature_dict, test_list, test_root)

    print(
        f"Test Model: {test_model_path}\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Threshold: {threshold:.3f}\n"
    )
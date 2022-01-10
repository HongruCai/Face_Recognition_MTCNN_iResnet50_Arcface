import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
import os
import csv

from Train_Test_Modules.MCTNN_Module.src.get_nets import PNet, RNet, ONet
from Train_Test_Modules.MCTNN_Module.src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from Train_Test_Modules.MCTNN_Module.src.first_stage import run_first_stage
from Train_Test_Modules.MCTNN_Module.src.visualization_utils import show_bboxes

pnet = PNet()
rnet = RNet()
onet = ONet()
onet.eval()

min_face_size = 15.0

thresholds = [0.6, 0.7, 0.8]

nms_thresholds = [0.7, 0.7, 0.7]


# f = open('photo.csv', 'w', encoding='utf-8', newline='' "")
# csv_writer = csv.writer(f)
# csv_writer.writerow(["photo_name", "x0", "y0", "x1", "y1"])

def get_train_cut():
    for filename in os.listdir('../train_set'):
        os.makedirs('../train_cut_result' + '/' + filename)
        os.makedirs('../test_set' + '/' + filename)
        rec = 0
        for filename1 in os.listdir('../train_set' + '/' + filename):
            print(filename1)
            image = Image.open('../train_set' + '/' + filename + '/' + filename1)
            width, height = image.size
            min_lenth = min(width, height)
            min_detection_size = 12
            factor = 0.707
            scales = []
            m = min_detection_size / min_face_size
            min_lenth *= m
            factor_count = 0
            while min_lenth > min_detection_size:
                scales.append(m * factor ** factor_count)  # 每次获得的图片都是原图片乘factor的n次方，n是迭代次数
                min_lenth *= factor
                factor_count += 1

            print('scales:', ['{:.2f}'.format(s) for s in scales])
            # print('number of different scales:', len(scales))

            # P-Net
            bounding_boxes = []
            # 不同的s上跑P-Net
            for s in scales:
                boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            bounding_boxes = np.vstack(bounding_boxes)
            # print('number of bounding boxes:', len(bounding_boxes))
            img1 = show_bboxes(image, bounding_boxes)
            # NMS + calibration

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
            img2 = show_bboxes(image, bounding_boxes)
            # R-Net
            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            with torch.no_grad():
                img_boxes = Variable(torch.FloatTensor(img_boxes))
            output = rnet(img_boxes)
            offsets = output[0].data.numpy()
            probs = output[1].data.numpy()
            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            img3 = show_bboxes(image, bounding_boxes)
            # MMS + calibration
            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
            img4 = show_bboxes(image, bounding_boxes)
            # O-Net
            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            with torch.no_grad():
                img_boxes = Variable(torch.FloatTensor(img_boxes))
            output = onet(img_boxes)
            landmarks = output[0].data.numpy()
            offsets = output[1].data.numpy()
            probs = output[2].data.numpy()
            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]
            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
            img5 = show_bboxes(image, bounding_boxes, landmarks)
            # NMS + calibration
            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]
            print('number of bounding boxes:', len(bounding_boxes))
            img6 = show_bboxes(image, bounding_boxes, landmarks)
            print()
            rec += 1
            if rec <= 3:
                for i in range(len(bounding_boxes)):
                    cropped = image.crop((int(bounding_boxes[i][0]), int(bounding_boxes[i][1]),
                                          int(bounding_boxes[i][2]), int(bounding_boxes[i][3])))
                    if i > 1:
                        cropped.save('../train_cut_result' + '/' + filename + '/' + filename1 + '/' + str(i))
                    else:
                        cropped.save('../train_cut_result' + '/' + filename + '/' + filename1)
            else:
                for i in range(len(bounding_boxes)):
                    cropped = image.crop((int(bounding_boxes[i][0]), int(bounding_boxes[i][1]),
                                          int(bounding_boxes[i][2]), int(bounding_boxes[i][3])))
                    if i > 1:
                        cropped.save('../test_set' + '/' + filename + '/' + filename1 + '/' + str(i))
                    else:
                        cropped.save('../test_set' + '/' + filename + '/' + filename1)


def get_cut(img_load_path, img_save_path, thers):
    image_ = Image.open(img_load_path)
    width, height = image_.size
    # print(width,height)

    min_lenth = min(width, height)

    min_detection_size = 12
    factor = 0.707

    scales = []

    m = min_detection_size / min_face_size
    # print(m)
    min_lenth *= m

    factor_count = 0

    # print(min_lenth)
    # print(min_detection_size)

    while min_lenth > min_detection_size:
        scales.append(m * factor ** factor_count)  # 每次获得的图片都是原图片乘factor的n次方，n是迭代次数
        min_lenth *= factor
        factor_count += 1

    print('scales:', ['{:.2f}'.format(s) for s in scales])
    print('number of different scales:', len(scales))

    # P-Net
    bounding_boxes = []
    # 不同的s上跑P-Net
    for s in scales:
        boxes = run_first_stage(image_, pnet, scale=s, threshold=thresholds[0])
        bounding_boxes.append(boxes)

    bounding_boxes = [i for i in bounding_boxes if i is not None]
    bounding_boxes = np.vstack(bounding_boxes)

    # print('number of bounding boxes:', len(bounding_boxes))

    # img1 = show_bboxes(image_, bounding_boxes)
    # img1.show()

    # NMS + calibration

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]

    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # img2 = show_bboxes(image_, bounding_boxes)
    # img2.show()
    print('P_Net processed')

    # R-Net
    img_boxes = get_image_boxes(bounding_boxes, image_, size=24)
    with torch.no_grad():
        img_boxes = Variable(torch.FloatTensor(img_boxes))
    output = rnet(img_boxes)
    offsets = output[0].data.numpy()
    probs = output[1].data.numpy()

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    # img3 = show_bboxes(image_, bounding_boxes)
    # img3.show()

    # MMS + calibration
    keep = nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    img4 = show_bboxes(image_, bounding_boxes)
    # img4.show()
    print('R_Net processed')

    # O-Net
    img_boxes = get_image_boxes(bounding_boxes, image_, size=48)
    # print(img_boxes)
    if len(img_boxes) == 0:
        return 0, bounding_boxes, 0
    else:
        with torch.no_grad():
            img_boxes = Variable(torch.FloatTensor(img_boxes))
        output = onet(img_boxes)
        landmarks = output[0].data.numpy()
        offsets = output[1].data.numpy()
        probs = output[2].data.numpy()

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        img5 = show_bboxes(image_, bounding_boxes, landmarks)
        # img5.show()
        print('O_Net processed')

        # NMS + calibration

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]
    print('number of bounding boxes:', len(bounding_boxes))

    # img6 = show_bboxes(image_, bounding_boxes, landmarks)
    # print(len(bounding_boxes))
    # print(bounding_boxes[0])
    # img6.show()
    # print(img6.size)
    rec = []
    for i_ in range(len(bounding_boxes)):
        if bounding_boxes[i_][2] - bounding_boxes[i_][0] < thers or bounding_boxes[i_][3] - bounding_boxes[i_][1] < thers:
            rec.append(i_)
    bounding_boxes = np.delete(bounding_boxes, rec, axis=0)
    landmarks = np.delete(landmarks, rec, axis=0)
    print('已忽略%-2d张人脸' % len(rec))
    print()
    for i in range(len(bounding_boxes)):
        cropped = image_.crop((int(bounding_boxes[i][0]), int(bounding_boxes[i][1]),
                               int(bounding_boxes[i][2]), int(bounding_boxes[i][3])))
        cropped.save(img_save_path + str(i) + '.jpg')
    return len(bounding_boxes), bounding_boxes, landmarks
import shutil
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from model.iResnet import iresnet50
from Train_Test_Modules.MCTNN_Module.MCTNN import get_cut
import cv2
import glob

# 用于标注的颜色库
color_lib = ['737979', '6d3a1d', '92272d', 'b93d47', 'e06168',
             'd6c9b3', 'a79297', '6b5258', 'bc3d36', 'e17e6b', 'efbea0',
             '58a557', '2d9b8e', '00285c', '000000',
             'fffbc9', '60a1d9', '547643', 'f9861c', 'C34A36',
             'd4d770', '70d781', '4eb1be', '2188a7', '4c5119', '19479d',
             'b7da8e', 'a5b45e'
             ]


def get_video_size(video_path):
    """获取视频宽高"""
    cap = cv2.VideoCapture(video_path)
    # print(cap.isOpened())
    if cap.isOpened():  # 正常打开
        rval, frame = cap.read()
    else:
        print('视频读取失败')
        return
    cv2.imwrite("temp.jpg", frame)
    img = Image.open('temp.jpg')
    w, h = img.size[0], img.size[1]
    img.close()
    os.remove('temp.jpg')
    # print(type(img.size))
    # print(img.size[0])
    # print(img.size[1])
    return w, h


def video2frames(video_src_path, frame_save_path, frame_width, frame_height, interval):
    """
    将视频按固定间隔读取写入图片
    :param video_src_path: 视频存放路径
    :param frame_save_path:　保存路径
    :param frame_width:　保存帧宽
    :param frame_height:　保存帧高
    :param interval:　保存帧间隔
    :return:　帧图片
    """
    each_video_save_full_path = os.path.join(frame_save_path) + "/"

    each_video_full_path = video_src_path

    cap = cv2.VideoCapture(each_video_full_path)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("读取失败!")

    while success:
        success, frame = cap.read()
        if success is False:
            print('视频处理结束')
            break
        print("正在读取第%d帧:" % frame_index, success)

        if frame_index % interval == 0:
            resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
            cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, resize_frame)
            frame_count += 1
        frame_index += 1
    cap.release()


def frames2video(fps, save_path, frames_path):
    """
    将处理后的图片重新组合为视频
    fps : 帧率
    save_path : 保存路径
    frames_path : 读取路径
    """
    f = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_path, f, fps, (width, height))
    imgs = glob.glob(frames_path + "/*.jpg")
    frame_num = len(imgs)
    for i in range(frame_num):
        print(i, "frame")
        if os.path.isfile("%s/%d.jpg" % (frames_path, i)):
            frame = cv2.imread("%s/%d.jpg" % (frames_path, i))
            videoWriter.write(frame)
    videoWriter.release()
    return


def cosin_metric(x1, x2):
    """计算两个特征间的余弦距离"""
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def pre_process(imgg_path, transf):
    """预处理图片"""
    img_ = Image.open(imgg_path)
    img_pr = transf(img_)
    # print(img_pr.shape)
    # print(img_pr.shape)
    img_pr.to(device)

    return img_pr


def verify(fearts, trans):
    """识别主函数"""
    sim = {}
    res = {}
    for label in os.listdir('facebank/'):
        sim[label] = []
        img_tensors = torch.empty(2, 3, 112, 112).to(device)
        for bankface in os.listdir('facebank/' + label):
            bankf_pre = pre_process('facebank/' + label + '/' + bankface, trans)
            img_tensors[0, :, :, :] = fearts
            img_tensors[1, :, :, :] = bankf_pre  # 将检测人脸与对比人脸拼接为一个tensor
            with torch.no_grad():
                feas = model(img_tensors)  # 输入网络
            feas1 = feas[0, :]
            feas2 = feas[1, :]  # 得到分别的特征
            feas1, feas2 = feas1.cpu().numpy(), feas2.cpu().numpy()
            similarity = cosin_metric(feas1, feas2)  # 计算特征的余弦距离
            if similarity < 0:
                similarity = -similarity  # 距离取绝对值
            # print(similarity)
            sim[label].append(similarity)  #
        # print(sim[label])
        temp = np.mean(sim[label])  # 计算label的距离平均值（在facebank中有多张图片的情况下）
        res[label] = temp
    # print(res)
    pred_ = max(res, key=lambda x: res[x])  # 找到距离最大，即概率最大的label
    ans = res[pred_]
    if res[pred_] < thres:  # 验证最大平均距离是否大于阈值，否则识别为unknown
        pred_ = 'unknown'

    return pred_, ans


def draw_box(img_path_, pred_, boxe, mark, dots_true):
    """对图片进行标注"""
    _img = Image.open(img_path_)
    draw = ImageDraw.Draw(_img, )

    for i_ in range(len(boxe)):
        x = (boxe[i_][0], boxe[i_][1])
        y = (boxe[i_][2], boxe[i_][3])
        if pred_[i_] != 'unknown':
            color = face_col[pred_[i_]]
        else:
            color = 'ffffff'
        draw.rectangle((x, y), outline='#' + color, width=int((1 / 30) * (boxe[i_][2] - boxe[i_][0])))  # 画出矩形
        # 画出人脸关键点
        if dots_true is True:
            mar = int((1 / 50) * (boxe[i_][2] - boxe[i_][0]))
            x1 = mark[i_][0] - mar
            y1 = mark[i_][5] - mar
            x1_ = mark[i_][0] + mar
            y1_ = mark[i_][5] + mar
            x2 = mark[i_][1] - mar
            y2 = mark[i_][6] - mar
            x2_ = mark[i_][1] + mar
            y2_ = mark[i_][6] + mar
            x3 = mark[i_][2] - mar
            y3 = mark[i_][7] - mar
            x3_ = mark[i_][2] + mar
            y3_ = mark[i_][7] + mar
            x4 = mark[i_][3] - mar
            y4 = mark[i_][8] - mar
            x4_ = mark[i_][3] + mar
            y4_ = mark[i_][8] + mar
            x5 = mark[i_][4] - mar
            y5 = mark[i_][9] - mar
            x5_ = mark[i_][4] + mar
            y5_ = mark[i_][9] + mar
            draw.chord((x1, y1, x1_, y1_), 0, 360, outline='#FF0000', width=max(int((1 / 80) * (boxe[i_][2] - boxe[i_][0])), 2))
            draw.chord((x2, y2, x2_, y2_), 0, 360, outline='#FF0000', width=max(int((1 / 80) * (boxe[i_][2] - boxe[i_][0])), 2))
            draw.chord((x3, y3, x3_, y3_), 0, 360, outline='#FF0000', width=max(int((1 / 80) * (boxe[i_][2] - boxe[i_][0])), 2))
            draw.chord((x4, y4, x4_, y4_), 0, 360, outline='#FF0000', width=max(int((1 / 80) * (boxe[i_][2] - boxe[i_][0])), 2))
            draw.chord((x5, y5, x5_, y5_), 0, 360, outline='#FF0000', width=max(int((1 / 80) * (boxe[i_][2] - boxe[i_][0])), 2))
        pos = (boxe[i_][0], boxe[i_][3] + 3)
        ft = ImageFont.truetype(font_path, int((1 / 6) * (boxe[i_][2] - boxe[i_][0])))  # 打上标签
        draw.text(pos, pred_[i_], fill='#' + color, font=ft)
    # _img.show()
    _img.save(res_save_path)  # 保存标注后的图片


if __name__ == '__main__':

    '''视频分帧'''
    video_load_path = 'face_input/origin_video/test_video_4.mp4'
    frames_save_path = 'face_input/frames/original'
    width, height = get_video_size(video_load_path)
    print('视频宽高为：%d,%d' % (width, height))
    internal = 1  # 视频采帧间隔
    thres = 50  # 小于此像素的人脸会被忽略
    video2frames(video_load_path, frames_save_path, width, height, internal)
    print()

    '''GPU使用'''
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU applied ')
    else:
        print('CPU applied')
        device = torch.device('cpu')

    '''预处理'''
    transform = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    '''载入模型'''
    model = iresnet50(dropout=0.0, fp16=True, num_features=512)
    model_path = 'model/params/ir50_glt.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    print(model_path + ' loaded')
    print()

    for ori_img in os.listdir('face_input/frames/original'):
        print('=' * 50)
        print('%s processing started...' % ori_img)
        print()
        '''MCTNN裁剪'''
        print('face detection started...')
        print()
        img_path = 'face_input/frames/original/' + ori_img  # 目标图片路径
        res_save_path = 'face_input/frames/processed/' + ori_img  # 保存处理后图片路径
        img_save_path = 'face_input/cut_res/'  # 保存人脸裁剪结果路径
        pre_cut_num, boxes, marks = get_cut(img_path, img_save_path, thres)
        print('face detection & cut ended')
        if pre_cut_num != 0:
            print('%d face(s) detected' % pre_cut_num)
            print()
        else:
            print('no face detected')  # 没有检测到人脸则将原图像保存
            shutil.copy(img_path, res_save_path)
            print('已将原图片复制')
            print()
            continue

        '''识别'''
        thres = 0.228
        pred = []
        print('recognition started')
        print()

        num = 0
        for im_g in os.listdir('face_input/cut_res'):
            img_pro = pre_process('face_input/cut_res/' + im_g, transform)
            res_v, pro = verify(img_pro, transform)  # 输入待处理的图片得到结果
            print('face %d label: %s  highest_ave_dis: %lf' % (num, res_v, float(pro)))
            num += 1
            pred.append(res_v)

        print()
        print('recognition ended')
        print()

        # 清理MCTNN生成的人脸剪切
        for i_mg in os.listdir('face_input/cut_res'):
            os.remove('face_input/cut_res/' + i_mg)
        print('cache cleared')
        print()

        '''图片人脸标注保存'''
        facebank_path = 'facebank/'
        face_col = {}
        k = 0
        for face_lists in os.listdir(facebank_path):
            face_col[face_lists] = color_lib[k % 31]
            k += 1
        font_path = 'model/Frick0.3-Regular-3.otf'
        draw_dots = True
        draw_box(img_path, pred, boxes, marks, draw_dots)
        print('%6s result generated' % ori_img)

    print('all frames processed, video generating')
    video_save_path = 'face_input/result/res.mp4'
    frames_read_path = 'face_input/frames/processed/'
    fps_num = 24  # 生成视频的帧率
    frames2video(fps_num, video_save_path, frames_read_path)
    print('video generated')

    # 清理中间图片
    for im in os.listdir('face_input/frames/original'):
        os.remove('face_input/frames/original/' + im)
    for im in os.listdir('face_input/frames/processed'):
        os.remove('face_input/frames/processed/' + im)

    print('cache cleared')
    print()
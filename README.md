# Face_Recognition_MTCNN_iResnet50_Arcface

基于MCTNN,iResNet50,Arcface的人脸识别与标注

Author : HenryTsai

## 项目介绍

- 目标为实现人脸识别(Face Recognition)
- 采用MTCNN & iResNet50 & ArcfaceLoss
- 实现了**图片**人脸识别标注与**视频**人脸识别标注
- 本项目仍有较多方面比较粗糙，优化与改进点很多，仅提供思路上的参考
- 如果有任何疑问，欢迎交流：
  
    E-mail: HenryTsai5683@outlook.com
    

## 文件框架

Face_Recognition_MTCNN_iResnet50_Arcface

├─Application_Pictures_Video—<Application:实现图片、视频标注 >

│  │  Picture_Recognition.py—<图片标注>

│  │  Video_Recognition.py—<视频标注>

│  ├─facebank—<人脸库>

│  ├─face_input—<存放原始图片或视频

│  └─model--<存放模型>

│      │  iResnet.py—<iResNet模型>

│      └─params—<存放模型参数>

└─Train_Test_Modules—<Train&Test:模型训练与测试 >

│  metrics.py— <ArcfaceLoss函数>

│  test_module.py—<模型测试>

│  train_module.py—<模型训练>

├─data—<存放测试、训练数据>

├─MTCNN_Module—<MTCNN模块>

│  │  MTCNN.py—<人脸定位与裁剪>

│  └─src—<MTCNN所需函数>

│      │  box_utils.py

│      │  detector.py

│      │  first_stage.py

│      │  get_nets.py

│      │  visualization_utils.py

│      │  **init**.py

│      └─weights—<MTCNN模型参数>

├─model--<存放模型>

│      fmobilenet.py—<MobileFaceNet模型>

│      iResnet.py—<iResNet模型>

└─params—<存放训练、测试模型参数>

## 模型训练

- 训练使用train_module.py
- 准备：
    - 数据集，可根据个人下载不同数据集，放入data中（需要jpg，其余格式可能报错）
    - 如果数据集中人脸并未分割，请自行调用MTCNN进行分割得到训练数据集
    - 修改train_module.py中的数据路径
    - 如需使用预训练参数，放入params，并修改模型路径
    - 其余包括batch_szie, optimizer, loss等，可自行修改
- 训练步骤：
    - 将图片输入网络得到特征
    - 特征与对应标签计算ArcfaceLoss，即余弦距离
    - 再根据loss函数计算loss
    - loss反向传播
- 训练结果：
    - 在小数据集上，loss可以降至0.5左右
    - 很容易发生过拟合
    - 如果出现nan，请检查你的数据，模型方面错误已排除
- 训练时并未关注acc，如有需要，可以自行添加

## 模型测试

- 测试使用test_module.py
- 测试方法为在LFW数据集上，对比两张人脸相似度
- 此处参考：
  
    [Build-Your-Own-Face-Model](https://github.com/siriusdemon/Build-Your-Own-Face-Model/blob/master/recognition/blog/test.md)
    
- 在其基础上，我修改为使用iResNet50测试，可以放入自己训练好的模型参数
- 测试结果：
    - 训练出的模型最优准确率95%左右
    - 采用[insight_face](https://github.com/deepinsight/insightface/tree/master/recognition)官方预训练模型可以达到98%

## 图片人脸标注

- 图片标注使用Picture_Recognition.py
- 准备：
    - 准备face_bank中的人脸，可以放入LFW数据集与需要检测的人脸
    - 在face_input中放入需要处理的图片
    - 修改文件中目标图片路径
    - params放入训练好的模型参数
    - 其余可以修改的参数：
        - 人脸大小：预设50，小于此像素大小的人脸会被MTCNN忽略，不进行标注
        - 距离阈值：预设0.228，为LFW上测试时得到的最佳距离，大于此距离才会认为是同一张人脸
        - 关键点描画：预设为True，会在标注人脸时标注其五个人脸关键点
    - 需要在model下放入标注所需的字体文件，并修改文件中的字体路径
- 标注步骤：
    - 调用MTCNN进行人脸定位与划分，将裁剪出的人脸储存
    - 对于裁剪出的每个人脸：
        - 与face_bank中每个标签下的人脸一起输入模型(iResNet50)，得到特征
        - 计算两张人脸间特征的余弦距离
        - 找到face_bank中距离最大的人脸对应的的标签（若face_bank中某一标签中有多张人脸，则取距离平均值）
        - 此标签为其预测出的人脸标签
        - 若最大距离小于设定的距离阈值，则认为face_bank中没有此人，输出标签为unknown
    - 所有人脸标签得到后，对原图片框出人脸，标注标签，打印人脸关键点
    - 输出图片
    - 清理中间文件
- 在我测试的图片中，都能很好的正确标注
- 同时，我设置了一些改进，边框粗细、字体大小都与人脸宽度成比例，来实现更好的标注效果

## 视频人脸标注

- 视频标注使用Video_Recognition
- 视频标注逻辑与图片相似，只是多出了先将视频一帧一帧分离，毎帧图片标注完成后，再将结果合成为新视频
- 一些文件中的路径需要自行修改，包括合成视频的帧数等
- 效果也算不错，视频分帧较多的话，时间可能较长
- 只尝试过mp4格式，其余格式可能报错

## Reference

本项目参考了以下仓库与文章，在此致谢

- [arcface_torch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
- [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
- [pytorch_arcface_cosface_partialFC](https://github.com/leoluopy/pytorch_arcface_cosface_partialFC)
- [手撕代码insightFace中的arcface_torch](https://zhuanlan.zhihu.com/p/368510746?utm_source=wechat_session&utm_medium=social&utm_oi=868793313396920320&utm_campaign=shareopn)
- [人脸识别合集 | 10 ArcFace解析](https://zhuanlan.zhihu.com/p/76541084)

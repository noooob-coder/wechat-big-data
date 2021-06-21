# **2021中国高校计算机大赛-微信大数据挑战赛Baseline**

本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。 

本次比赛以多个行为预测结果的加权uAUC值进行评分。大赛官方网站：https://algo.weixin.qq.com/
baseline仅仅包含模型代码不包含特征抽取代码
## **1. 环境配置**

- pandas 1.0.5
- pytorch 1.6
- python 3

## **2. 运行配置**

- gpu-1080ti	 
- cpu-i7-7800X
- 内存-16g

## **3. 目录结构**
- 数据存放于data目录下
- data_load.py 用于数据的读取以及预处理
- deep_fm.py 存放deepfm模型
- featureSlect.py 利用lgb模型对特征进行筛选
- lgb.py lgb模型

## **4. 运行方式**
python main.py 
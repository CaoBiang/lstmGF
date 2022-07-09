# AiGNSS
Predict GNSS observation data through Deep Learning and Machine Learning. 

通过机器学习和深度学习对GNSS观测值进行预测，当然包括一些组合观测值。

Actually, this project is my undergraduate graduation project. Due to my learning ability problem, I spent several months on ML/DL and GNSS data processing, hoping to record it through GitHub. But let's be honest: this is an extremely shallow and naive experiment.

其实这个项目是我的本科毕业设计，由于自身学习能力问题，花费数月精力在学习ML/DL与GNSS数据处理上，希望通过GitHub进行记录留念。但说实话：这只是一个极其浅显且幼稚的实验。

## GF二次差序列
说实话效果不错，根据有时候误差可以达到0.013m以内。

## 为何不使用原始观测数据？
对于此问题，我与导师进行了探讨，有两个原因：
### 1.原始数据稳定性差，组合观测值可以消除部分误差。
### 2.原始观测数据有强趋势性/自相关性，LSTM预测效果极差。
载波相位观测值原始数据预测效果图：

![image](/images/1.jpg)

参考博客：

[1]https://blog.csdn.net/youhuakongzhi/article/details/114552592

[2]https://ask.csdn.net/questions/1084891

## 网络的设计
网络由输入层、隐含层与输出层构成。在本次设计中，输入层神经元的个数l就是用来预测下一历元数据的前l个历元，也就是时间步。输入层的数据被输入到LSTM层之中，经过m个LSTM节点，在最后一个节点处输出m维的向量。LSTM层输出向量到n个节点的Dense层中，经过Dense层的维度变换，使LSTM层输出的向量在输出层变换为1维标量，也就是下一个历元的预测值。
![image](/images/2.jpg)

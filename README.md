#AiGNSS
Predict GNSS observation data through Deep Learning and Machine Learning. 

通过机器学习和深度学习对GNSS观测值进行预测，当然包括一些组合观测值。

Actually, this project is my undergraduate graduation project. Due to my learning ability problem, I spent several months on ML/DL and GNSS data processing, hoping to record it through GitHub. But let's be honest: this is an extremely shallow and naive experiment.

其实这个项目是我的本科毕业设计，由于自身学习能力问题，花费数月精力在学习ML/DL与GNSS数据处理上，希望通过GitHub进行记录留念。但说实话：这只是一个极其浅显且幼稚的实验。

##为何不使用原始观测数据？
对于此问题，我与导师进行了探讨，有两个原因：
### 1.原始数据稳定性差，组合观测值可以消除部分误差。
### 2.原始观测数据有强趋势性/自相关性，LSTM预测效果极差。
载波相位观测值原始数据预测效果图：

![image](/images/1.jpg)

综合考虑，使用二次差值来进行预测，本次使用GF组合观测值二次差值。

参考博客：

[1]https://blog.csdn.net/youhuakongzhi/article/details/114552592

[2]https://ask.csdn.net/questions/1084891

##网络的设计
网络由输入层、隐含层与输出层构成。在本次设计中，输入层神经元的个数l就是用来预测下一历元数据的前l个历元，也就是时间步。输入层的数据被输入到LSTM层之中，经过m个LSTM节点，在最后一个节点处输出m维的向量。LSTM层输出向量到n个节点的Dense层中，经过Dense层的维度变换，使LSTM层输出的向量在输出层变换为1维标量，也就是下一个历元的预测值。

![image](/images/2.jpg)

##数据集的构建
使用Pandas库进行对不同历元的数据依次进行滑动窗口分割。将i至i+l-1历元的数据组合为一个训练集的输入值，也就是一个样本，存放于二维矩阵train_X；同时i+l历元的数据作为这个训练集的目标值，存放于二维矩阵train_Y。同理，在训练好模型之后，若要使用模型进行预测，则只需要输入二维矩阵predict_X，经过运算后输出predict_Y。

![image](/images/3.jpg)

每次训练所使用的训练集train_X由很多个样本组成，构建成一个(feature_num,time_step,sample_size)的三维矩阵，才能成为LSTM层的输入。其中feature_num为特征数量，time_step为时间步，sample_size为样本数量，由于特征只有一种观测值，所以三维矩阵只有1页。

![image](/images/4.jpg)

##效果预览

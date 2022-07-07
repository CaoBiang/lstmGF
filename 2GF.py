# https://github.com/CaoBiang/AiGNSS

import pandas as pd
import numpy as np
import re
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,LSTM
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from matplotlib.axis import Axis
import tensorflow as tf

data=pd.read_csv("data.csv")
#归一化数据
value_num=data["2GF"].values
value_num=np.delete(value_num,[0,1])#去掉前几个数据
#print(value_num)
max_num=np.max(value_num)
min_num=np.min(value_num)
value_num_normalize_data=(value_num-min_num)/(max_num-min_num)
print(max_num,min_num,max_num-min_num)
#增加一个维度，表示每个时间点只有一个特征
value_num_normalize_data=np.expand_dims(value_num_normalize_data,axis=1)
#print(value_num_normalize_data)

#构造数据
#生成训练集
#设置常量
time_step=5      #时间步
input_size=1      #输入层维度
output_size=1     #输出层维度
train_x_t,train_y_t=[],[]   #训练集
for i in range(len(value_num_normalize_data)-time_step):
    x=value_num_normalize_data[i:i+time_step]
    y=value_num_normalize_data[i+time_step:i+time_step+1][0]
    train_x_t.append(x.tolist())  #将数组转化成列表
    train_y_t.append(y.tolist())
train_y_t=[i[0]for i in train_y_t]
#print(train_y_t)

PRE=[]
TEST_Y2=[]
l=40#用多少组数据进行训练
times=50
for i in range(times):
    model = Sequential()
    model.add(LSTM(units=256, input_shape=(time_step, 1)))
    model.add(Dense(units=256))
    model.add(Dense(1))
    model.compile(loss=["mae"], optimizer='adam',metrics=["mae"])
    train_x=train_x_t[i:i+l]
    train_y=train_y_t[i:i+l]
    test_x=train_x_t[i+l:i+l+1]
    test_y=train_y_t[i+l:i+l+1]
    history = model.fit(np.array(train_x), np.array(train_y), epochs=200,batch_size=10,verbose=0,shuffle=True)
  #预测测试集,并转为原始数据
    pre=model.predict(np.array(test_x))
    pre=[p[0]*(max_num-min_num)+min_num for p in pre ]
  #将测试集转换为原始值
    test_y2=[p*(max_num-min_num)+min_num for p in test_y ]
    
    PRE.append(pre[0])
    TEST_Y2.append(test_y2[0])
    print(i,pre[0],'\t',test_y2[0])
#print(PRE)
#print(TEST_Y2)

#绘制图
pyplot.figure(figsize=(5,3),dpi=300)
#pyplot.title('Predicted Value Versus Actual Value')
pyplot.plot(PRE, label='预测值')
pyplot.plot(TEST_Y2, label='真实值')
pyplot.legend()
pyplot.show()

#神经网络残差
delta=[]
for i in range(len(PRE)):
    delta.append(abs(TEST_Y2[i]-PRE[i]))
deltaSorted=sorted(delta)

yaxis=[0.01,0.02,0.03,0.04,0.05]
#yaxis.append(np.mean(delta))
yaxis.append(np.max(delta))
pyplot.figure(figsize=(5,3),dpi=300).add_subplot().yaxis.set_ticks(yaxis)
#pyplot.title('Residual of Predicted Value')
pyplot.bar(range(len(delta)),delta,label='残差')
#pyplot.axvline(24, color='r', linestyle='--', label='加入周跳的历元')
pyplot.xlabel('历元')
pyplot.ylabel('单位:米')
#pyplot.axhline(y=np.mean(delta),color="blue",label='Mean')
print('L:',l,'STEP:',time_step,'MAX:',np.max(delta),'MIN:',np.min(delta),'MEAN:',np.mean(delta))
pyplot.legend()
pyplot.show()

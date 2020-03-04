from sklearn import svm
import random
import numpy as np
from matplotlib import pyplot as plt
import PointFeatureCompute

import os

data = np.loadtxt("./treeAndRoad_Fv.txt",dtype = float,delimiter=',')
# svm training
x,y=np.split(data,indices_or_sections=(2,),axis=1) #打乱

clf = svm.SVC(kernel='linear') #创建一个svm分类器
clf.fit(x,y)  #喂入数据
print(clf)

#读入待分类数据
Pred_data=np.loadtxt('./points/test/test1.txt')
print('测试集: ',Pred_data.shape)
Pred_data_=np.zeros((Pred_data.shape[0],3),float)
Pred_data_[:,:]=Pred_data[:,0:3]


#新建结果点云
f=open('./PointCloudLabel.txt','a')

#计算特征向量
fv_Pred=PointFeatureCompute.GetFeatureVector(Pred_data_)
print('测试集的特征向量: ',fv_Pred.shape)
p0=fv_Pred[0,:]


#预测每一个点
for j in range(0,fv_Pred.shape[0]):

    label = clf.predict([fv_Pred[j,:]])  #预测点云中每一个点 ，这里要加一个中括号，否则会报错参考：https://blog.csdn.net/qq_41185868/article/details/79007557?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
    print('label: ', label)
    if label == 1:
        f.writelines(str(Pred_data_[j,0])+" "+str(Pred_data_[j,1])+" "+str(Pred_data_[j,2])+" 0 "+"255 "+"0"+"\n")
    else:
        f.writelines(str(Pred_data_[j,0])+" "+str(Pred_data_[j,1])+" "+str(Pred_data_[j,2])+" 255 "+"0 "+"0"+"\n")



    print('完成进度：',j/fv_Pred.shape[0]*100,'%')
f.close()

# #预测每一个点
#
#
# label = clf.predict([p0])  #预测点云中每一个点
# print('label: ',label)
# f.writelines(str(Pred_data_[0,0])+" "+str(Pred_data_[0,1])+" "+str(Pred_data_[0,2])+' '+str(label)+"\n")
# # print('完成进度：',j/fv_Pred.shape[0]*100,'%')
# f.close()

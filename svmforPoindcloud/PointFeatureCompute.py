'''
计算输入点云的每一个点的特征向量
'''

#1.读入点云
#2 为每一个点计算k邻域
#3 计算邻域内的协方差矩阵
#4 计算曲率、粗糙度、发散状指数、面状指数、法向量与z轴夹角、邻域内高度差（邻域特征）
#5 保存为一个n*F的矩阵 ，n为点数，f为特征数

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

#计算邻域内的点距离拟合平面的距离
def CaculateAverageSquareDistance(pointMat):  # 输入 ppointMat为一个矩阵，返回距离的平方和
    num = pointMat.shape[0]
    B = np.zeros((pointMat.shape[0],3))
    one = np.ones((pointMat.shape[0],1))
    B[:,0] = pointMat[:,0]
    B[:,1] = pointMat[:,1]
    B[:,2] = one[:,0]
    l = pointMat[:,2]
    BTB = np.matmul(B.T,B)
    BTB_1 = np.linalg.pinv(BTB)
    temp = np.matmul(BTB_1,B.T)
    result = np.matmul(temp,l)
    V  = np.matmul(B,result)-l
    sum = 0
    for i in range (0,V.shape[0]):
        sum = sum+V[i]**2
    return sum/V.shape[0]


#计算粗糙度
def CaculateRoughness(point,pointMat):  #输入point 应该是一个点,pointMat是这个点云,返回一个值，代表该点邻域的粗糙度


    neigh = NearestNeighbors(n_neighbors=13)   #最近的20个点
    neigh.fit(pointMat)
    index = neigh.kneighbors([point],return_distance=False) #z最近邻的点的索引
    avedis2 = CaculateAverageSquareDistance(pointMat[index].reshape(13,3))   #每一个点邻域的表面粗超度


    return avedis2*10000


# 邻域分析计算发散状指数
def NeiAna(point,pointMat):  #输入：点云，点；输出：该点的发散状指数
    p1 = pointMat[:, 0:3]


    neigh = NearestNeighbors(n_neighbors=13)
    neigh.fit(pointMat)
    index = neigh.kneighbors([point], return_distance=False)

    pp = p1[index].reshape(13, 3)  #指定点的邻域点云
    pp[:, 0] = pp[:, 0] - np.mean(pp[:, 0])
    pp[:, 1] = pp[:, 1] - np.mean(pp[:, 1])
    pp[:, 2] = pp[:, 2] - np.mean(pp[:, 2])
    a = pp.T  # 矩阵转置
    b = np.matmul(pp.T, pp)  # 协方差矩阵
    fev = np.linalg.eigvals(b)  # 特征值
    fev = (np.sort(fev))  # 特征值排序，从小到大
    scattering= (fev[0]) / fev[2] * 1000  # λ3/λ1 (λ3<λ2<λ1)  计算发散指数

    return scattering


# 组成特征向量【发散状指数，粗糙度】
# 输入：点云p,输出：nx2的特征矩阵
def GetFeatureVector(p):
    fv=np.zeros((p.shape[0],2),float)
    for i in range(p.shape[0]):

        fv[i, 0] = NeiAna(p[i],p) #邻域分析计算发散状指数
        fv[i, 1] = CaculateRoughness(p[i],p) #粗糙度
    return fv

if __name__ == '__main__':
    #计算tree的特征
    p = np.loadtxt("./points/tree/tree1.txt" )
    pointIn=np.zeros((p.shape[0],3),float)
    pointIn[:,:]=p[:,0:3]

    print(pointIn.shape)
    fv=GetFeatureVector(pointIn)
    print(fv.shape)


    #保存Fv
    type=1 #树的类型设置为1
    f = open('treeAndRoad_Fv.txt', 'a')
    for i in range(fv.shape[0]):

        f.write(str(fv[i,0])+","+str(fv[i,1])+","+str(type)+"\n")
    f.close()


# -----------------------------------------------------------------------------------------

# 计算road的特征
    p_road = np.loadtxt("./points/road/road3.txt" )
    pointIn=np.zeros((p_road.shape[0],3),float)
    pointIn[:,:]=p_road[:,0:3]

    print(pointIn.shape)
    fv_road=GetFeatureVector(pointIn)
    print(fv_road.shape)

    #保存Fv
    type=-1 #road的类型设置为-1
    f = open('treeAndRoad_Fv.txt', 'a')
    for i in range(fv_road.shape[0]):

        f.write(str(fv_road[i,0])+","+str(fv_road[i,1])+","+str(type)+"\n")
    f.close()

    #可视化特征
    plt.scatter(fv[:, 0], fv[:, 1], c='g')#树红色 type 为1
    plt.scatter(fv_road[:,0],fv_road[:,1],c='r') #路红色 type 为-1
    plt.title('Features')
    plt.show()
# -*- coding: utf-8 -*-
# @Date    : 2015/7/21  21:57
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : 
# @Description:玩的Perceptron，数据集用的是台大机器学习课后作业1的数据

import numpy as np
import math as mt
import re

#从PLA_data.dat中取出数据。
#输入参数 filename:list     输出参数 data_set:list; label_set:list
def GetData(filename):
    with open(filename) as file_opened:
        data_lines=file_opened.readlines()
        data_set=[]
        label_set=[]
        for line in data_lines:
            line_splited=re.split(' |\t',line.strip())
            data_set.append([float(num) for num in line_splited[0:(len(line_splited)-1)]])
            label_set.append(float(line_splited[-1]))
        return data_set,label_set

#找出那些分错的数据点
#输入参数 data_set_matrix:matrix; label_set_list:list w:matrix    输出参数 data_set_matrix[i]:matrix; label_set_list[i]:int
def FindWrongPoint(data_set_matrix,label_set_list,w):
    for i in range(data_set_matrix.shape[0]):
        if label_set_list[i]*(data_set_matrix[i]*w.T)<=0:
            return True,data_set_matrix[i],label_set_list[i]
    return False,data_set_matrix[i],label_set_list[i]

def PlaTrain(data_set_list,label_set_list,eta):
    data_set_matrix=np.column_stack((np.mat(data_set_list),np.mat([1 for i in xrange(len(data_set_list))]).T))

    w=np.mat(np.tile(0,(1,data_set_matrix.shape[1])))   #生成参数向量，扩充到K+1维了,w为1*(K+1)维向量
    while(True):
        flag,wrong_point,label_of_wrong_point=FindWrongPoint(data_set_matrix,label_set_list,w)
        if flag==True:
            w=w+eta*label_of_wrong_point*wrong_point
            print w
        else:
            return w


if __name__ == '__main__':
    data_set_list,label_set_list=GetData('PLA_data.dat')
    print PlaTrain(data_set_list,label_set_list,1)


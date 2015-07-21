# -*- coding: utf-8 -*-
# @Date    : 2015/7/21  21:57
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : 
# @Description:

import numpy as np
import math as mt
import re

def GetData(filename):
    with open(filename) as file_opened:
        data_lines=file_opened.readlines()
        data_set=[]
        label_set=[]
        for line in data_lines:

            line_splited=re.split(' |\t',line.strip())
            data_set.append(line_splited[0:(len(line_splited)-1)])
            label_set.append(line_splited[-1])
        return data_set,label_set

def PlaTrain(data_set_list,label_set_list):
    data_set_matrix=np.column_stack((np.mat(data_set_list),np.mat([1 for i in xrange(len(data_set_list))]).T))
    label_set_matrix=np.mat(label_set_list).T

    w=np.mat(np.tile(0,(1,data_set_matrix.shape[1])))


if __name__ == '__main__':
    data_set_list,label_set_list=GetData('PLA_data.dat')
    PlaTrain(data_set_list,label_set_list)


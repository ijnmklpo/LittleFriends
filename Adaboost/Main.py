# -*- coding: utf-8 -*-
# @Author: ijnmklpo
# @Date:   2016-03-10 19:36:03
# @Last Modified by:   ijnmklpo
# @Last Modified time: 2016-03-11 14:07:02
# @Desc:


import DataSet_IO_Module as DS_IO_Module
import Adaboost_Module
import numpy as np


if __name__=='__main__':
	#data_set,label_set=DS_IO_Module.GetDataFromCSV('F:/csci527_hw2_training.csv')

	# data_set=[[0,1,3],[0,3,1],[1,2,2],[1,1,3],[1,2,3],
	# 		  [0,1,2],[1,1,2]]		#,[1,1,1],[1,3,1],[0,2,1]
	# label_set=[-1,-1,-1,-1,-1,-1,1]	#,1,-1,-1
	# 
	data_set=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
	label_set=[1,1,1,-1,-1,-1,1,1,1,-1]
	claasifier_list=Adaboost_Module.AdaboostMain(data_set,label_set,3)

	DS_IO_Module.OutputDataToCSV(claasifier_list,'F:/csci527_hw2_classifiers.csv')

	test_data=[1,1,1]
	classifier_right_vector=np.mat((np.array(claasifier_list)).T[4])
	predicted_label_list=[]
	for classifier in claasifier_list:
		if test_data[classifier[0]]==classifier[1]:
			predicted_label_list.append(classifier[2])
		else:
			predicted_label_list.append(classifier[3])
	print predicted_label_list
	print classifier_right_vector*(np.mat(predicted_label_list)).T
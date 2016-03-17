# -*- coding: utf-8 -*-
# @Author: ijnmklpo
# @Date:   2016-03-10 19:36:03
# @Last Modified by:   ijnmklpo
# @Last Modified time: 2016-03-11 14:07:02
# @Desc:


import time

import numpy as np

import Adaboost_Module

if __name__=='__main__':
	start_time=time.time()

	# pseudo dataset. I roughly identify that test data points [1,2,3],[1,3,4],[1,4,4] belong\
	# to label 1, the result is 100% correct. But if test data points [4,1,2], [4,2,3], it\
	# classifies them as label 3, maybe it's not a good conclusion, while I identified [4,2,3]\
	# as label 2.
	data_set=[[1,3,1],[1,4,2],[4,4,3],[5,3,4],[3,3,4],
			  [2,1,4],[1,2,1],[1,1,1],[4,4,3],[1,1,2]]
	label_set=[1,2,2,3,3,3,1,1,2,1]


	claasifier_list=Adaboost_Module.AdaboostMain(data_set,label_set,50)


	print 'training time:',time.time()-start_time
	test_data=[4,1,3]
	classifier_right_vector=np.mat((np.array(claasifier_list)).T[4]).tolist()[0]
	print classifier_right_vector
	predicted_label_list=[]
	for classifier in claasifier_list:
		if test_data[classifier[0]]==classifier[1]:
			predicted_label_list.append(classifier[2])
		else:
			predicted_label_list.append(classifier[3])
	print predicted_label_list
	score_dict=dict()
	for right,pred_label in zip(classifier_right_vector,predicted_label_list):
		if score_dict.has_key(pred_label):
			score_dict[pred_label]+=right
		else:
			score_dict[pred_label]=right
	highest_score=0
	highest_label=-1
	print score_dict
	for label,right in score_dict.items():
		if right>highest_score:
			highest_label=label
			highest_score=right
	print 'predict ',test_data,' as:',highest_label,' with score:',highest_score
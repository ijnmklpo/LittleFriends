# -*- coding: utf-8 -*-
# @Author: ijnmklpo
# @Date:   2016-03-10 19:35:40
# @Last Modified by:   ijnmklpo
# @Last Modified time: 2016-03-10 20:21:41
# @Desc:


PERC=0.9

import csv
import numpy as np

def GetDataFromCSV(filename):
	'''
	'''
	print 'Start getting data from"'+filename+'"...'
	data_features_set=[]
	data_label_set=[]
	with file(filename,'rb') as csv_file_opened:
		csv_reader=csv.reader(csv_file_opened)
		for index,line in enumerate(csv_reader):
			if index==0:
				continue
			data_features=[int(item) for item in line[:-1]]
			data_features_set.append(data_features)
			data_label_set.append(int(line[-1]))
		print 'Finish getting data.'
		return data_features_set,data_label_set
			

def OutputDataToCSV(data_set,filename):
	print 'Start output classifiers to"'+filename+'"...'
	with file(filename,'wb') as csv_file_opened:
		writer = csv.writer(csv_file_opened)
		writer.writerows(data_set)
		csv_file_opened.close()
		print 'Finish outputing.'


def GetDataFromCsvAsValid(filename):
	'''
	'''
	print 'Get validat data from"'+filename+'"...'
	data_set=[]
	train_data_set=[]
	train_label_set=[]
	valid_data_set=[]
	valid_label_set=[]
	with file(filename,'rb') as csv_file_opened:
		csv_reader=csv.reader(csv_file_opened)
		for index,line in enumerate(csv_reader):
			if index==0:
				continue
			data_set.append(line)
		np.random.shuffle(data_set)
		train_data=data_set[0:int(PERC*len(data_set))]
		valid_data=data_set[int(PERC*len(data_set)):]

		for line in train_data:
			train_data_set.append([int(item) for item in line[:-1]])
			train_label_set.append(int(line[-1]))

		for line in valid_data:
			valid_data_set.append([int(item) for item in line[:-1]])
			valid_label_set.append(int(line[-1]))

		print 'Finish cut data.'
		return train_data_set,train_label_set,valid_data_set,valid_label_set
# -*- coding: utf-8 -*-
# @Author: ijnmklpo
# @Date:   2016-03-10 19:35:40
# @Last Modified by:   ijnmklpo
# @Last Modified time: 2016-03-10 20:21:41
# @Desc:


import csv


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
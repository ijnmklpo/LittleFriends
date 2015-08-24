# -*- coding: utf-8 -*-
# @Date    : 2015/8/4  9:42
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : 
# @Description:Kaggle-San Francisco Crime Classification. Processing both in Naive Bayes and SoftMax(Logistic regression in
#               multiple classes),and by this way we gain the two kinds of probability matrix,and I want to combine these two
#               probabilities as scores and determine the terminal probability for every term(use the score as pseudo
#               probability)

import numpy as np
import math as mt
import csv


#-----------------------------------------------------------------------------
def PreprocessTrainData(filename_in,filename_out):
    data_set_list=[]
    with file(filename_in, 'rb') as file_opened:
        reader = csv.reader(file_opened)
        flag=True
        for line in reader:
            if flag==True:
                flag=False
                continue
            data=[]
            column=line[0].split(' ')
            column_one=column[0].split('/')
            column_two=column[1].split(':')
            data.append(float(column_one[0]))
            data.append(float(column_one[1]))
            data.append(float(column_one[2]))
            data.append(float(column_two[0])*60+float(column_two[1]))
            data.extend(line[1:6])
            data.append(float(line[7]))
            data.append(float(line[8]))
            data_set_list.append(data)

    with file(filename_out, 'wb') as file_opened:
        writer = csv.writer(file_opened)
        writer.writerows(data_set_list)


def GetTrainData(filename):
    print "Start getting raw data from csv file..."
    with file(filename, 'rb') as file_opened:
        data_set_list=[]
        label_list=[]
        reader=list(csv.reader(file_opened))
        for line in reader:
            data=[]
            # print line[0]
            # print type(line[0])
            data.append(float(line[0]))
            data.append(float(line[1]))
            data.append(float(line[2]))
            data.append(float(line[3]))
            data.extend(line[5:8])
            data.append(float(line[9]))
            data.append(float(line[10]))

            data_set_list.append(data)
            label_list.append(line[4])
        print "Finish data getting!"
        return data_set_list,label_list


def GetDataWithValidation(filename):
    print "Start getting raw data from csv file..."
    with file(filename, 'rb') as file_opened:
        train_data_set_list=[]
        train_label_list=[]
        test_data_set_list=[]
        test_label_list=[]
        reader=np.random.permutation(list(csv.reader(file_opened)))
        data_set1_list=reader[0:int(0.9*len(reader))]
        data_set2_list=reader[int(0.9*len(reader)):len(reader)]
        for line in data_set1_list:
            data=[]
            # print line[0]
            # print type(line[0])
            data.append(float(line[0]))
            data.append(float(line[1]))
            data.append(float(line[2]))
            data.append(float(line[3]))
            data.extend(line[5:8])
            data.append(float(line[9]))
            data.append(float(line[10]))

            train_data_set_list.append(data)
            train_label_list.append(line[4])
        for line in data_set2_list:
            data=[]
            # print line[0]
            # print type(line[0])
            data.append(float(line[0]))
            data.append(float(line[1]))
            data.append(float(line[2]))
            data.append(float(line[3]))
            data.extend(line[5:8])
            data.append(float(line[9]))
            data.append(float(line[10]))

            test_data_set_list.append(data)
            test_label_list.append(line[4])
        print "Finish data getting!"
        return train_data_set_list,train_label_list,test_data_set_list,test_label_list
#-----------------------------------------------------------------------------

def CreateVocabulary(nominal_data_list,test_data_list,label_list):
    print "Start creating vocabularies of nominal data and labels..."
    feature_vocabulary_list=[]
    label_vocabulary_list=[]
    for line in nominal_data_list:
        for item in line:
            if item not in feature_vocabulary_list:
                feature_vocabulary_list.append(item)
    for line in test_data_list:
        for item in line:
            if item not in feature_vocabulary_list:
                feature_vocabulary_list.append(item)

    for item in label_list:
        if item not in label_vocabulary_list:
            label_vocabulary_list.append(item)
    print "Finish Create Vocabularies!"
    return feature_vocabulary_list,label_vocabulary_list


def TrainBayes(nominal_data_list,label_list,feature_vocabulary_list,label_vocabulary_list):
    print "Start training Naive Bayes step:"
    #print len(feature_vocabulary_list),len(label_vocabulary_list)
    feature_given_label_conditional_prob_matrix=np.ones((len(label_vocabulary_list),len(feature_vocabulary_list)))
    #print feature_given_label_conditional_prob_matrix.shape
    label_prob_matrix=np.ones((1,len(label_vocabulary_list)))   #本来应该是zeros，这里做了一个不是很妥当的平滑，有可能会使所谓概率超过1

    count=0
    for label in label_list:
        label_prob_matrix[0,label_vocabulary_list.index(label)]+=1
        feature_given_label_conditional_prob_matrix[label_vocabulary_list.index(label),feature_vocabulary_list.index(nominal_data_list[count][0])]+=1
        feature_given_label_conditional_prob_matrix[label_vocabulary_list.index(label),feature_vocabulary_list.index(nominal_data_list[count][1])]+=1
        feature_given_label_conditional_prob_matrix[label_vocabulary_list.index(label),feature_vocabulary_list.index(nominal_data_list[count][2])]+=1
        count+=1

    feature_given_label_conditional_prob_matrix=np.log(feature_given_label_conditional_prob_matrix)-np.log(np.tile((label_prob_matrix.T),(1,feature_given_label_conditional_prob_matrix.shape[1])))
    label_prob_matrix=np.log(label_prob_matrix/(label_prob_matrix.sum()))
    print "Training step finish!"
    return feature_given_label_conditional_prob_matrix,label_prob_matrix


def BayesClassify(test_data_transformed_matrix,feature_given_label_conditional_prob_matrix,label_prob_matrix):
    print "Start classification step..."
    print test_data_transformed_matrix.shape
    print feature_given_label_conditional_prob_matrix.shape
    whole_prob_list=[]
    for item in test_data_transformed_matrix:
        category_vector=np.array(item*(feature_given_label_conditional_prob_matrix.T)+label_prob_matrix).tolist()[0]
        #print category_vector
        whole_prob_list.append(category_vector)
    return whole_prob_list
    #whole_prob_matrix=test_data_transformed_matrix*(feature_given_label_conditional_prob_matrix.T)+np.mat(np.tile(label_prob_matrix,(len(test_data_transformed_matrix),1)))
    #return whole_prob_matrix



#把测试数据从list转换成向量组（一个向量构成的矩阵），每行都是一个事务的特征向量
def TestDataTransform(nominal_test_data_list,feature_vocabulary_list):
    test_data_transformed_list=[]
    for line in nominal_test_data_list:
        #print line
        test_feature_vector_list=[0 for i in xrange(len(feature_vocabulary_list))]
        test_feature_vector_list[feature_vocabulary_list.index(line[0])]+=1
        test_feature_vector_list[feature_vocabulary_list.index(line[1])]+=1
        test_feature_vector_list[feature_vocabulary_list.index(line[2])]+=1
        test_data_transformed_list.append(test_feature_vector_list)
    return test_data_transformed_list

def NaiveBayes(data_set_list,label_list,test_data_list,test_label_list):
    print "Get into Naive Bayes Process:"
    nominal_data_list=[]
    nominal_test_data_list=[]
    for line in data_set_list:
        nominal_data_list.append(line[4:7])
    for line in test_data_list:
        nominal_test_data_list.append(line[4:7])    #在validation的时候和真正使用的时候取的范围是不一样的
    feature_vocabulary_list,label_vocabulary_list=CreateVocabulary(nominal_data_list,nominal_test_data_list,label_list)
    test_data_transformed_list=TestDataTransform(nominal_test_data_list,feature_vocabulary_list)    #在测试阶段使用
    feature_given_label_conditional_prob_matrix,label_prob_matrix=TrainBayes(nominal_data_list,label_list,feature_vocabulary_list,label_vocabulary_list)    #训练好的模型用于分类
    whole_label_prob_list=BayesClassify(np.mat(test_data_transformed_list),feature_given_label_conditional_prob_matrix,label_prob_matrix)

    predicted_label_list=[]
    count=0.0
    for item in whole_label_prob_list:
        item_ndarray=np.array(item)
        predicted_label_list.append(label_vocabulary_list[(item_ndarray-max(item_ndarray)).tolist().index(0)])
    print len(predicted_label_list),len(test_label_list)
    for i in xrange(len(predicted_label_list)):
        if predicted_label_list[i]==test_label_list[i]:
            count+=1
    print count/len(predicted_label_list)


if __name__ == '__main__':
    #PreprocessTrainData('E:/Kaggle/Kaggle1/train.csv','E:/Kaggle/Kaggle1/train_processed.csv')
    #data_set_list,label_list=GetTrainData('E:/Kaggle/Kaggle1/train_processed.csv')
    #NaiveBayes(data_set_list,label_list)
    train_data_set_list,train_label_list,test_data_set_list,test_label_list=GetDataWithValidation('E:/Kaggle/Kaggle1/train_processed.csv')
    NaiveBayes(train_data_set_list,train_label_list,test_data_set_list,test_label_list)

# -*- coding: utf-8 -*-
# @Author: ijnmk
# @Date:   2016-03-10 00:08:42
# @Last Modified by:   ijnmklpo
# @Last Modified time: 2016-03-11 14:36:50
# @Desc:	This is the code of adaboost(adapt boosting).Have finished the adaboost feature for nominal only
# 		data and numarical only data respectively, and it supports multi-class classifing. but the framework
# 		of this code is terrible. the main API is 
# 								def AdaboostMain(data_set,label_set,classifier_num)
# 		If we want to process specific type data(either nominal or numerical),it is necessary to switch the
# 		code blocks in "AdaboostMain(data_set,label_set,classifier_num)" to corresponding components.nominal
# 		data corresponds to "def IsOrNotDecisionStump" and "def IsOrNotDataRightUpdate" while numerical data 
# 		corresponds to "def LessOrGreaterDecisionStump" and "def LessOrGreaterDataRightUpdate".
# 			Later I will reognize this code, maybe put the two parts into a class whose name is 'Adaboost',
# 		and merge the two features to get a adaboost model which can process both nominal data and numerical
# 		data. Well,hope I can remenber it.
# 		

import numpy as np
import math as mt

ALPHA=0.5	#在计算弱分类器权值的时候用于权衡权值的计算（如果ALPHA很大，则如果出现了一个分类很准确的分\
			#类器，其权重会非常大，若出现一个分类很糟糕的分类器，其权值会非常小。因此ALPHA小的话会更平滑）
STEP_NUM=20.0	#在

		

#----------------------------------------------------------------------------------------------------------<block>
#主要给外部提供的接口

def AdaboostMain(data_set,label_set,classifier_num):
	'''
	Desc:Adaboost的主程序入口，传入训练集，训练集的标签集，以及规定弱分类器的数量
	Param:
		- data_set:list[list[]]。	内层list为每个数据点的特征，外层list即包含了所有数据点。
		- label_set:list[]。		所有数据的标签列表。
		- classifier_num:int。		该Adaboost模型中含有的弱分类器数量。
	Return:
		- claasifier_list:list[list[]]。	内层每个list即是decision stump，是一个5维向量，第1个元素是\
											所选的特征，第2个元素是所选的当前特征的取值，第3个元素是跟\
											当前特征取值相同的数据被分为什么label，第4个元素是与当前特\
											征取值不同的数据被分为什么leibel，第5个元素是当前弱分类器\
											的权值。						
	'''
	print 'Start adaboost step:'
	claasifier_list=[]	#所有弱分类器都放这里
	data_num=len(data_set)
	data_right_vector=[1.0/data_num for i in xrange(data_num)]	#初始化数据权值
	for i in xrange(classifier_num):
		print 'iteration:'+str(i+1)+'---------------------'

		#可替换模块-----------------------------------------------
		error_with_right,classifier=LessOrGreaterDecisionStump(data_set,label_set,data_right_vector)
		#---------------------------------------------------------

		classifier_right=ALPHA*(mt.log((1-error_with_right))-mt.log(error_with_right))
		classifier.append(classifier_right)
		claasifier_list.append(classifier)

		#可替换模块-----------------------------------------------
		data_right_vector=LessOrGreaterDataRightUpdate(data_set,label_set,data_right_vector,classifier)
		#---------------------------------------------------------
		print 'Finish this iteration!--------------------'
	print 'Finish adaboost training!'
	return claasifier_list


#-------------------------------------------------------------------------------------------------------</block>



#-------------------------------------------------------------------------------------------------------<block>
#实现Adaboost模块的主体代码

def IsOrNotDecisionStump(data_set,label_set,data_ratio_vector):
	'''
	Desc:标称数据分类器，是该Adaboost的弱分类器，本块代码可以替换为其他分类模型。当前代码是用的CART树桩的分\
			类器。其思路首先遍历所有特征，然后内层遍历所有特征的取值，根据当前特征取值将数据点划分为特征是\
			该取值和不是其取值两类，然后分别统计两个类别中的数据点所属最多的类别作为当前数据点的类别，这样\
			就构成了一个弱的树桩分类器，最后根据这个分类结果计算出其带权误差和。两层循环结束后选出带权误差和\
			最低的那个作为当前迭代所选的若分类器。\
	Parameters：
		- data_set:list[list[]]。	内层list为每个数据点的特征，外层list即包含了所有数据点。
		- label_set:list[]。		所有数据的标签列表。
		- data_ratio_vector:list[float]。	每个数据点当前迭代的权值。
	Return：
		- best_score:float。	选出的当前弱分类器最优的带权分类误差。
		- best_result:list。	当前弱分类器的信息，是一个四维向量。第1个元素是所选的特征，第2个元素是所选\
								的当前特征的取值，第3个元素是跟当前特征取值相同的数据被分为什么label，第4个\
								元素是与当前特征取值不同的数据被分为什么leibel\					
	'''
	print 'Start creating stump...'
	data_set=np.array(data_set)
	feature_num=len(data_set[0])
	feature_dict=[]
	#首先构造出所有特征取值的词典用于后续做分类循环时取值。
	for feature_index in xrange(feature_num):
		feature_dict.append(list(set(data_set.T[feature_index])))

	best_result=[None,None,None,None]
	best_score=None
	for feature_index in xrange(feature_num):
		cur_feature_vector=np.mat(data_set.T[feature_index])
		for feature in feature_dict[feature_index]:
			'''
			decision code(using CART tree idea) and chose a best feature and its cut point by computing the\
			inpurity with right.
			'''
			classify_vect=(np.array(cur_feature_vector.A==feature)).tolist()[0]
			cur_score,label_1,label_0=ComputeInpurityWithRight(classify_vect,label_set,data_ratio_vector)
			if cur_score<best_score or best_score==None:
				best_result=[feature_index,feature,label_1,label_0]
				best_score=cur_score
	
	'''
	Here we return the best feature and its cut point(if the data has the same value to the cut point in \
	corresponding feature), this can be regard as a decision stump,looks like a stump with only one branch
	'''
	print 'Finish stump creating!'
	return best_score,best_result


def ComputeInpurityWithRight(classify_vect,label_set,data_ratio_vector):
	'''
	Desc:计算用当前特征的当前取值进行划分的带权误差率。该函数首先是要计算当前特征某取值进行划分后两个划分\
			子空间分别该属于什么label，这个通过统计各自子空间中数据所属的label，然后以数据点加权求和最大的\
			label作为该子空间的label，因为这样可以使当前子空间分类误差率最低。\
			
	Param:
		- classify_vect:list[boolean]。数据当前特征是否是属于某个取值，所以其中每个元素都是boolean型变量。
		- label_set:list[int]。数据对应的标签类别list。
		- data_ratio_vector:list[float]。其实也就是data_right_vector，元素是当前迭代中每个数据的权重。
	Return:
		- final_score:float。由当前特征的某个取值做划分计算出来的总带权误差率。
		- highest_label_in_isdata:int。该弱分类器划分后数据当前特征值等于所选特征的数据集中所判定的类别。
		- highest_label_in_isnotdata:int。该弱分类器划分后数据当前特征值不等于所选特征的数据集中所判定的类别。
	'''

	final_score=0.0
	#做两个子空间中每个label得分的统计
	is_data_count=dict()	#特征值属于当前取值的数据统计量
	isnot_data_count=dict()	#特征值不属于当前取值的数据统计量
	for index,data_point in enumerate(classify_vect):
		if data_point==True:
			if is_data_count.has_key(label_set[index]):
				is_data_count[label_set[index]]+=data_ratio_vector[index]
			else:
				is_data_count[label_set[index]]=data_ratio_vector[index]
		else:
			if isnot_data_count.has_key(label_set[index]):
				isnot_data_count[label_set[index]]+=data_ratio_vector[index]
			else:
				isnot_data_count[label_set[index]]=data_ratio_vector[index]

	#分别找出两个子空间中得分最高的label
	highest_count_in_isdata=0.0
	highest_label_in_isdata=None
	highest_count_in_isnotdata=0.0
	highest_label_in_isnotdata=None
	for key,value in is_data_count.items():
		if value>highest_count_in_isdata:
			highest_count_in_isdata=value
			highest_label_in_isdata=key
	for key,value in isnot_data_count.items():
		if value>highest_count_in_isnotdata:
			highest_count_in_isnotdata=value
			highest_label_in_isnotdata=key

	#计算该分类器最终的总带权误差率
	for feature_flag,label,data_ratio in zip(classify_vect,label_set,data_ratio_vector):
		if feature_flag==True:
			if label!=highest_label_in_isdata:
				final_score+=data_ratio
		else:
			if label!=highest_label_in_isnotdata:
				final_score+=data_ratio
	'''
	另一种总带权误差率计算方式（向量运算）：
	classify_vect_mat=np.mat(classify_vect)
	label_vect=np.mat(label_set)
	data_ratio_vect_mat=np.mat(data_ratio_vector)
	final_score=((np.multiply(classify_vect_mat,label_vect))==highest_label_in_isdata)*data_ratio_vect_mat.T+\
			((np.multiply((classify_vect_mat+1)%2,label_vect))==highest_label_in_isnotdata)*data_ratio_vect_mat.T
	'''
	return final_score,highest_label_in_isdata,highest_label_in_isnotdata


def IsOrNotDataRightUpdate(data_set,label_set,cur_data_right_vector,classifier):
	'''
	Desc:该函数用于计算当迭代结束后（生成了一个弱分类器）每个数据点的新权值。
	Param:
		- data_set:list[list[int]]。数据特征集。
		- label_set:list[int]。数据的label集。
		- cur_data_right_vector:list[float]。未更新前数据点的权值向量。
		- classifier:list[int]。当前弱分类器信息向量。是一个5维向量，第1个元素是\
								所选的特征，第2个元素是所选的当前特征的取值，第3个元素是跟\
								当前特征取值相同的数据被分为什么label，第4个元素是与当前特\
								征取值不同的数据被分为什么leibel，第5个元素是当前弱分类器\
								的权值。						
	Return:
		- updated_data_right_vector:list[]。更新后数据权重向量。
	'''
	print 'Start updating the data rights...'
	updated_data_right_vector=[]	#更新后的数据权值会放在该list中。
	cur_feature_data_line=np.array(data_set).T[classifier[0]]	#找出该弱分类器用于分类的那列特征向量。

	#根据公式更新每个数据点的权值。
	for feature,label,cur_data_right in zip(cur_feature_data_line,label_set,cur_data_right_vector):
		if feature==classifier[1]:
			updated_data_right_vector.append(cur_data_right*mt.exp(-classifier[4]*mt.pow(-1,(label!=classifier[2]))))
		else:
			updated_data_right_vector.append(cur_data_right*mt.exp(-classifier[4]*mt.pow(-1,(label!=classifier[3]))))
	updated_data_right_sum=sum(updated_data_right_vector)
	updated_data_right_vector=[item/updated_data_right_sum for item in updated_data_right_vector]
	print 'Finish updating data rights...'
	return updated_data_right_vector


#------------------------------------------------------------------------------------------------------------</block>


#------------------------------------------------------------------------------------------------------------<block>
#
def LessOrGreaterDecisionStump(data_set,label_set,data_ratio_vector):
	'''
	Desc:数值数据分类器，是该Adaboost的弱分类器。该代码仍然是用的CART树桩分类器。\
			\
	Parameters：
		- data_set:list[list[]]。	内层list为每个数据点的特征，外层list即包含了所有数据点。
		- label_set:list[]。		所有数据的标签列表。
		- data_ratio_vector:list[float]。	每个数据点当前迭代的权值。
	Return：
		- best_score:float。	选出的当前弱分类器最优的带权分类误差。
		- best_result:list。	当前弱分类器的信息，是一个四维向量。第1个元素是所选的特征，第2个元素是所选\
								的当前特征的取值，第3个元素是跟当前特征取值相同的数据被分为什么label，第4个\
								元素是与当前特征取值不同的数据被分为什么leibel\					
	'''
	print 'Start creating stump...'
	data_set=np.array(data_set)
	feature_num=len(data_set[0])
	cut_point_lists=[]
	#首先构造出所有特征取值的词典用于后续做分类循环时取值。
	for feature_data_list in data_set.T:
		feature_max=feature_data_list.max()
		feature_min=feature_data_list.min()
		step_len=(feature_max-feature_min)/STEP_NUM
		cut_point_lists.append(np.arange(feature_min,feature_max,step_len))

	best_result=[None,None,None,None]
	best_score=None
	for feature_index in xrange(feature_num):
		cur_feature_vector=np.mat(data_set.T[feature_index])
		for cut_point in cut_point_lists[feature_index]:
			'''
			decision code(using CART tree idea) and chose a best feature and its cut point by computing the\
			inpurity with right.
			'''
			classify_vect=(cur_feature_vector.A>cut_point).tolist()[0]
			cur_score,label_1,label_0=ComputeInpurityWithRight(classify_vect,label_set,data_ratio_vector)
			if cur_score<best_score or best_score==None:
				best_result=[feature_index,cut_point,label_1,label_0]
				best_score=cur_score
	
	'''
	Here we return the best feature and its cut point(if the data has the same value to the cut point in \
	corresponding feature), this can be regard as a decision stump,looks like a stump with only one branch
	'''
	print 'Finish stump creating!'
	return best_score,best_result


def LessOrGreaterDataRightUpdate(data_set,label_set,cur_data_right_vector,classifier):
	'''
	Desc:该函数用于计算当迭代结束后（生成了一个弱分类器）每个数据点的新权值。
	Param:
		- data_set:list[list[int]]。数据特征集。
		- label_set:list[int]。数据的label集。
		- cur_data_right_vector:list[float]。未更新前数据点的权值向量。
		- classifier:list[int]。当前弱分类器信息向量。是一个5维向量，第1个元素是\
								所选的特征，第2个元素是所选的当前特征的取值，第3个元素是跟\
								当前特征取值相同的数据被分为什么label，第4个元素是与当前特\
								征取值不同的数据被分为什么leibel，第5个元素是当前弱分类器\
								的权值。						
	Return:
		- updated_data_right_vector:list[]。更新后数据权重向量。
	'''
	print 'Start updating the data rights...'
	updated_data_right_vector=[]	#更新后的数据权值会放在该list中。
	cur_feature_data_line=np.array(data_set).T[classifier[0]]	#找出该弱分类器用于分类的那列特征向量。

	#根据公式更新每个数据点的权值。
	for feature,label,cur_data_right in zip(cur_feature_data_line,label_set,cur_data_right_vector):
		if feature>classifier[1]:
			updated_data_right_vector.append(cur_data_right*mt.exp(-classifier[4]*mt.pow(-1,(label!=classifier[2]))))
		else:
			updated_data_right_vector.append(cur_data_right*mt.exp(-classifier[4]*mt.pow(-1,(label!=classifier[3]))))

	#做归一化
	updated_data_right_sum=sum(updated_data_right_vector)
	updated_data_right_vector=[item/updated_data_right_sum for item in updated_data_right_vector]
	print 'Finish updating data rights...'
	return updated_data_right_vector
#--------------------------------------------------------------------------------------------------------------
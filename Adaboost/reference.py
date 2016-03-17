# -*- coding: utf-8 -*-
# @Author: ijnmklpo
# @Date:   2016-03-11 10:10:04
# @Last Modified by:   ijnmklpo
# @Last Modified time: 2016-03-11 11:21:03
# @Desc:不是我的

from __future__ import division
import numpy as np
import scipy as sp

class WEAKC:
	def __init__(self,X,y):
		'''
			X is a N*M matrix
			y is a length M vector
			M is the number of traincase
			this weak classifier is a decision Stump
			it's just a basic example from <统计学习方法>
		'''
		self.X=np.array(X)		#数据特征list
		self.y=np.array(y)		#标签list
		self.N=self.X.shape[0]	#特征数量
	def train(self,W,steps=100):
		'''
			W is a N length vector
		'''
		#print W
		min = 100000000000.0	
		t_val=0;	
		t_point=0;	
		t_b=0;	
		self.W=np.array(W)	#W是当前特征的权重向量
		for i in range(self.N):
			q ,err = self.findmin(i,1,steps)	#arg中，i是第i个特征
			if (err<min):
				min = err
				t_val = q
				t_point = i
				t_b = 1		#t_b直观地可以看做如果某个数据第i个特征的值大于q，那么就判定为类别1
		for i in range(self.N):
			q ,err = self.findmin(i,-1,steps)
			if (err<min):
				min = err
				t_val = q
				t_point = i
				t_b = -1	#t_b直观地可以看做如果某个数据第i个特征的值大于q，那么就判定为类别-1
		#昨晚后就得到了一个弱分类器----------
		self.t_val=t_val
		self.t_point=t_point
		self.t_b=t_b
		#------------------------------------
		print self.t_val,self.t_point,self.t_b
		return min
	def findmin(self,i,b,steps):	#i：	b：
		#这段相当于初始化-------------------------------------
		t = 0
		now = self.predintrain(self.X,i,t,b).transpose()
		err = np.sum((now!=self.y)*self.W)
		#print now
		##---------------------------------------------------
		pgd=0
		buttom=np.min(self.X[i,:])
		up=np.max(self.X[i,:])
		mins=1000000;
		minst=0
		st=(up-buttom)/steps
		for t in np.arange(buttom,up,st):	#取若干个特征取值作为划分点，而并不是选遍所有特征之间的空隙作为划分点
			now = self.predintrain(self.X,i,t,b).transpose()
			#print now.shape,self.W.shape,(now!=self.y).shape,self.y.shape
			err = np.sum((now!=self.y)*self.W)	#选定弱分类器的标准还是带权误差，所以在多类别问题中，还是要以选定\
												#特征、取值、两个子空间类别后的带权误差和作为分类器性能标准
			if err<mins:
				mins=err
				minst=t
		return minst,mins
	def predintrain(self,test_set,i,t,b):
		'''
		该函数根据特征i以阈值为t判断出每个数据为正或为负，如果参数b为+1则阈值大于t为1；若参数b为-1则阈值小于t为1
		'''
		test_set=np.array(test_set).reshape(self.N,-1)	#这个-1参数相当于自动对齐这个矩阵。这个矩阵中每一列才是\
														#一个数据，因为self.N是特征数。
		gt = np.ones((np.array(test_set).shape[1],1))	#产生label列表
		#print np.array(test_set[i,:]*b)<t*b
		gt[test_set[i,:]*b<t*b]=-1	#若b=1，则把第i个特征判断小于t的判断为-1；\
									#若b=-1，则把第i个特征判断大于t的判断为-1
									#这个代码是把两个划分的子空间到底选什么label（+1或-1）分两次判断，所以可以看\
									#到最外面有两个相同的循环，只是参数b的值不一样，这两个循环做的是相同的工作，\
									#只是划分后的子空间所选的label相反了而已。
		return gt

	def pred(self,test_set):
		test_set=np.array(test_set).reshape(self.N,-1)
		t = np.ones((np.array(test_set).shape[1],1))
		t[test_set[self.t_point,:]*self.t_b<self.t_val*self.t_b]=-1
		return t
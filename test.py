# -*- coding: utf-8 -*-
# @Date    : 2015-03-31 15:30:39
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import numpy as np
import math as mt

if __name__=='__main__':
	P=np.mat([[0.65,0.28,0.07],[0.15,0.67,0.18],[0.12,0.36,0.52]])
	x=np.mat([0.21,0.68,0.11])
	for i in range(30):
		x=x*P
		print x

	a=1
	b=2



	print x*P[:,0]


	print x[0,0]
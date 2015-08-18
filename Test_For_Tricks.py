# -*- coding: utf-8 -*-
# @Date    : 2015/7/24  10:01
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : 
# @Description:This is the test py file,for testing the usage of python

import numpy as np
import math as mt

if __name__ == '__main__':
    a=[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    aa=np.mat(a)
    print len(aa[0,:]),len(aa)
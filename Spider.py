# -*- coding: utf-8 -*-
# @Date    : 2015-04-21 16:21:55
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version :
# @Description :随便玩玩的基本爬虫

import numpy as np
import math as mt
import re		#regularization
import urllib as spider

#获得整个html页面内容
#参数  urlname:string
def GetHTML(urlname):
	HTMLopened=spider.urlopen(urlname)
	webPage=HTMLopened.read()
	return webPage		#string：输出整个页面内容

#通过re正则表达式，从html文件中截出匹配的内容
#参数  urlname:string
def GetIMG(htmlStr):
	imgRe=re.compile(r'src="(.+?\.png)" a')	#好像是匹配整个string，但是只取出()中的内容，其中?是表示使用非贪婪模式
											#compile的结果是把str类型的正则表达式转换成一个pattern
	imgGet=imgRe.findall(webPage)		#findall()是遍历整个字串，找出所有匹配的子串然后输出成一个list
	return imgGet		#list：是匹配到的每个片段组成的list

if __name__=='__main__':
	webPage=GetHTML("http://www.cnblogs.com/fnng/p/3576154.html")
	imgGet=GetIMG(webPage)
	for i in range(len(imgGet)):
		spider.urlretrieve(imgGet[i],'D:/%i.png' %i)		#如果路径不写只有文件名则会放在工程目录下，用这种嵌入法可以生成动态的字符串
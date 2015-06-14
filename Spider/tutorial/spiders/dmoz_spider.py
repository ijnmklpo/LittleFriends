# -*- coding: utf-8 -*-
# @Date    : 15-6-12  下午4:43
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : 
# @Discription :dmoz.org的爬虫

import scrapy

class DmozSpider(scrapy.Spider):
    name='dmoz'
    start_urls=["http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
                "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/"]
    def parse(self,response):
        filename=response.url.split("/")[-2]
        with open(filename,'wb') as f:     #with语句挺有趣，它相当于是一个try的异常处理块
            f.write(response.body)
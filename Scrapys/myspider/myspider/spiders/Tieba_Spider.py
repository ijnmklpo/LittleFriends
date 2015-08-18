# -*- coding: utf-8 -*-
# @Date    : 2015/7/23  10:21
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : 
# @Description:简单的爬虫，从贴吧爬某个用户发的图片，用Request把1~26页的图片链接都存在items.json中，如果
#              想下的话可以写一个算法用urllib把图片下过来，具体下载代码参照Spider.py

import scrapy as sc
from scrapy.http import Request
from myspider.items import Tieba_Item

class Tieba_Spider(sc.Spider):
    name='tieba_spider'
    #allowed_domains=['http://tieba.baidu.com']
    start_urls=['http://tieba.baidu.com/p/2356000394?pn=1']

    def parse(self,response):
        i=0
        for sel in response.xpath('//div[@class="l_post l_post_bright j_l_post clearfix"]'):
            item=Tieba_Item()
            if(sel.xpath('div[@class="d_author"]/ul/li[@class="d_name"]/@data-field').extract()[0]==u'{"user_id":675601164}'):
                #item['user_name']=sel.xpath('div[@class="d_author"]/ul/li[@class="d_name"]/a/text()').extract()
                item['image']=sel.xpath('div[@class="d_post_content_main "]/div[@class="p_content "]/cc/div/img/@src').extract()
                yield item
        sel=response.xpath('//div[@class="p_thread thread_theme_7"]/div/ul/li')
        print "This is the",sel.xpath('span/text()').extract()[0],"page"
        #print 1111111
        #item['page']=sel.xpath('span/text()').extract()[0]
        #yield item
        for fragment in sel.xpath('a'):
            if fragment.xpath('text()').extract()[0]==u'\u4e0b\u4e00\u9875':
                yield Request('http://tieba.baidu.com'+fragment.xpath('@href').extract()[0])
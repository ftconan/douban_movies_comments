# -*- coding: utf-8 -*-
import scrapy


class MovieCommentSpider(scrapy.Spider):
    name = 'movie_comment'
    allowed_domains = ['douban.com']
    start_urls = ['http://douban.com/']

    def parse(self, response):
        pass

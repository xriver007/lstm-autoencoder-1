from urlparse import urlparse
import os.path
import json
from datetime import datetime
# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class HTMLPipeline(object):
    HTML_DATA_DIR = './data'

    def process_item(self, item, spider):
        if not os.path.exists(self.HTML_DATA_DIR):
            os.mkdir(self.HTML_DATA_DIR)

        with open(os.path.join(
                self.HTML_DATA_DIR, 
                self.filename(item)), 'w+') as f:
            f.write(item['body'][0].encode('utf-8'))
        return item

    def filename(self, item):
        url = item['url'][0]
        domain = urlparse(url).netloc
        domain = domain.replace('.', ',')
        salt = abs(hash(url))
        length = len(item['body'][0])
        return '{}-{}-{}'.format(domain, salt, length)

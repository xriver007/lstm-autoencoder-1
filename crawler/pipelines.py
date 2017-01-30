# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
try:
    from urlparse import urlparse
except:
    from urllib.parse import urlparse
import os.path
import json
from datetime import datetime


class HTMLPipeline(object):
    HTML_DATA_DIR = './data'

    def process_item(self, item, spider):
        if not os.path.exists(self.HTML_DATA_DIR):
            os.mkdir(self.HTML_DATA_DIR)

        with open(os.path.join(
                self.HTML_DATA_DIR, 
                self._filename(item)), 'w+') as f:
            f.write(item['body'][0])
        return item

    def _filename(self, item):
        url = item['url'][0]
        domain = urlparse(url).netloc
        domain = domain.replace('.', ',')
        salt = abs(hash(url))
        length = len(item['body'][0])
        return '{}-{}-{}'.format(domain, salt, length)

from . import BaseSpider


class HTMLSpider(BaseSpider):
    name = 'html'
    start_urls = [
        'https://en.wikipedia.org/wiki/Main_Page',
        'https://techcrunch.com/'
    ]

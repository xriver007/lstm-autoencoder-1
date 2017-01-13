from scrapy.loader import ItemLoader
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from ..items import HTMLItem


class BaseSpider(CrawlSpider):
    rules = (
        Rule(LinkExtractor(), callback='parse_body', follow=True),
    )

    def parse_body(self, response):
        loader = ItemLoader(item=HTMLItem(), response=response)
        loader.add_value('url', response.url)
        loader.add_css('body', 'body')
        return loader.load_item()

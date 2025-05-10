import scrapy
from scrapy_playwright.page import PageMethod
from datetime import datetime
import re

class MyntraSpider(scrapy.Spider):
    name = "myntra_spider"
    allowed_domains = ["myntra.com"]
    
    default_product_url = "https://www.myntra.com/tshirts/flying+machine/flying-machine-floral-printed-relaxed-fit-t-shirt/31690009/buy"

    custom_settings = {
        'DOWNLOAD_DELAY': 5,
        'CONCURRENT_REQUESTS': 1,
        'LOG_LEVEL': 'WARNING',
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'ITEM_PIPELINES': {'__main__.MyntraPipeline': 300},
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            'headless': True,
            'timeout': 30 * 1000,
            'proxy': None 
        },
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.myntra.com/',
        }
    }

    def __init__(self, product_urls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if product_urls:
            if isinstance(product_urls, str):
                self.product_urls = [product_urls]
            else:
                self.product_urls = product_urls
        else:
            self.product_urls = [self.default_product_url]

    def start_requests(self):
        for url in self.product_urls:
            yield scrapy.Request(
                url,
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        PageMethod("wait_for_load_state", "networkidle"),
                        PageMethod("wait_for_selector", "div.pdp-details", timeout=10000),
                        PageMethod("wait_for_selector", "div.pdp-price-info", timeout=10000)
                    ],
                    "playwright_include_page": True
                },
                errback=self.errback,
                callback=self.parse_product
            )

    def parse_product(self, response):
        if "Oops! Something went wrong" in response.text:
            self.logger.warning("Bot detected by Myntra!",response.url)
            return

        image_style = response.xpath("//div[contains(@class, 'image-grid-image')]/@style").get()
        image_url = re.search(r'url\("(.+?)"\)', image_style).group(1) if image_style else None
        
        price = response.xpath('''
            (//span[contains(@class, 'pdp-price')]/strong/text() |
             //span[contains(@class, 'pdp-discount')]/preceding-sibling::span/text())[1]
        ''').get("").replace("₹", "").strip()

        item = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "myntra.com",
            "product_url": response.url,
            "product_name": response.xpath("(//a[contains(@class, 'breadcrumbs-link')])[last()-1]/text()").get("").strip(),
            "brand": response.xpath("//h1[contains(@class, 'pdp-title')]/text()").get("").strip(),
            "category": response.xpath("(//a[contains(@class, 'breadcrumbs-link')])[last()-2]/text()").get("").strip(),
            "description": response.xpath("//h1[contains(@class, 'pdp-name')]/text()").get("").strip(),
            "price": price,
            "material": response.xpath("//div[contains(text(), 'Fabric')]/following-sibling::div/text()").get("").strip(),
            "image_url": image_url,
            "scraping_success": True
        }
        
        if not item["price"]:
            item["scraping_success"] = False
            self.logger.warning(f"Failed to extract price from {response.url}")

        yield item

    async def errback(self, failure):
        self.logger.error(f"Request failed: {failure.value}")
        if failure.check(scrapy.exceptions.IgnoreRequest):
            yield {
                "product_url": failure.request.url,
                "scraping_success": False,
                "error": str(failure.value)
            }

class MyntraPipeline:
    items = []
    
    def open_spider(self, spider):
        MyntraPipeline.items.clear()
    
    def process_item(self, item, spider):
        MyntraPipeline.items.append(item)
        return item

if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    
    process = CrawlerProcess(settings={
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "DOWNLOAD_HANDLERS": {
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        }
    })
    
    product_urls = [
        "https://www.myntra.com/tshirts/the+souled+store/the-souled-store-pennywise-printed-round-neck-cotton-oversized-t-shirt/28682758/buy",
        "https://www.myntra.com/tshirts/levis/levis-men-black-printed-round-neck-pure-cotton-t-shirt/2363452/buy"
    ]
    
    process.crawl(MyntraSpider, product_urls=product_urls)
    process.start()
    
    print(f"Scraped {len(MyntraPipeline.items)} products:")
    for idx, item in enumerate(MyntraPipeline.items, 1):
        print(f"\nProduct {idx}:")
        for key, value in item.items():
            print(f"{key.upper()}: {value}")

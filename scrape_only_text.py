# https://github.com/Facico/Chinese-Vicuna/tree/master
import requests
import os
from bs4 import BeautifulSoup, SoupStrainer
import re
#from trafilatura import fetch_url, extract
#from trafilatura.utils import sanitize, trim
import os
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import json
import time
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers.utils import logging
import asyncio

logging.set_verbosity_error()

DATA_PATH = 'data/text_unprocessed.json'
URL_PATH = 'data/urls_unprocessed.json'

BLACK_LIST_LANG = [ '/cn/', '/kr/', '/ru/', '/jp/', '/tw/', '/po/', '/th/', '/fr/', '/de/', '/es/',  '/br/', '/cn', '/kr', '/ru', '/jp', '/tw', '/th', '/fr', '/de', '/es',  '/br', '/trade/search', '/privacy', '/economy', 'facebook', 'instagram', 'youtube', 'twitch', 'discord', 'reddit', 'tiktok', 'linkedin', 'pinterest', 'tumblr', 'vimeo', 'snapchat' ]
alpha = re.compile(r'[a-zA-Z]')
counter = 0

lock = threading.Lock()
black_list_url = set()
todo_list_url = set()

class Crawler:
    
    def __init__(self, url: str, max_depth: int):

        self.url = url
        self.base_url = url
        self.max_depth = max_depth
        self.depth = 0

        options = Options() 
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1280,1280")
        options.add_argument("--no-sandbox")
        options.add_argument("--enable-javascript")

        self.driver = webdriver.Chrome(options=options)
        
        self.driver.implicitly_wait(2)

    def crawl(self, url: str, depth=0):
        global counter
        max_depth = self.max_depth
        base_url = self.base_url

        if depth > max_depth:
            return
        
        with lock:
            if url in black_list_url or url in todo_list_url or any([lang in url for lang in BLACK_LIST_LANG]):
                return
        
        if any(item in url for item in ['pathofexile.com', base_url]):
            pass
        else:
            return

        self.driver.get(url)
        time.sleep(0.25)
        html = self.driver.page_source

        if html is None:
            return
        
        with lock: # probably not needed
            todo_list_url.add(url)

        # get links on the page
        to_visit = []
        soup = BeautifulSoup(html, 'html.parser')
        a_list = set([urljoin(url, node.get('href')) for node in soup.find_all('a')])

        i = 0
        for a in a_list:
            if a not in to_visit and a not in black_list_url:
                i += 1
                to_visit.append(a)
                self.crawl(a, depth=depth+1)
        
        soup_extra = BeautifulSoup(html, 'html.parser')
        remove_list = { 'Payment', 'Buy ', 'preferences', 'partners', 'partner', 'paypal' ,'skrill', 'crypto', 'facebook', 'instagram', 'privacy', 'policy', 'creative', 'commons', 'disclaimers', 'patreon', 'twitter', 'youtube', 'twitch', 'discord', 'reddit', 'tiktok', 'linkedin', 'pinterest', 'tumblr', 'vimeo', 'snapchat', 'whatsapp', 'imprint', 'impressum', 'contact', 'cookies', 'privacy', 'policy', 'legal', 'data', 'protection', 'sitemap', 'accessibility', 'accessable', 'access', 'help', 'faq', 'support', 'donate', 'donation', 'donations', 'report', 'reports', 'advertising', 'cookies', 'cookie'}
        time_save = time.time()
        # Remove header
        try:
            header = soup_extra.find('header')
            header.decompose()
        except:
            pass

        try:
            footer = soup_extra.find('footer')
            footer.decompose()
        except:
            pass

        try:
            nav = soup_extra.find('nav')
            for n in nav:
                n.decompose()
        except:
            pass

        try:
            sidebars = soup_extra.find_all('aside')
            for sb in sidebars:
                sb.decompose()
        except:
            pass

        for tag in soup_extra.find_all('a'):
            if any(x in tag.get_text().lower() for x in remove_list):
                tag.decompose()

        url_text = soup_extra.get_text('\n', strip=True)
        
        if url_text is None:
            return

        url_text = ' '.join([ x for x in TextBlob(url_text).raw_sentences if ('' if any(y in x.lower() for y in remove_list) else x) ])
        
        url_text = url_text.replace('\n', ' ')
        url_text = re.sub(r'[^\sa-zA-Z0-9\._-]', '', url_text)
        url_text = re.sub(r'([A-Z])', r' \1', url_text)
        url_text = re.sub('\s+', ' ', url_text)
        url_text = re.sub(r'[^\w\s]', '', url_text).strip()

        url_text = url_text.replace(' Po E ', ' PoE ')
        url_text = url_text.replace(' Po E.', ' PoE.')
        url_text = url_text.replace(' Po E,', ' PoE,')
        url_text = url_text.replace('Po E ', 'PoE ')

        process_delta = time.time() - time_save
        if url_text == '':
            return
        
        if not alpha.match(url_text):
            return

        tmpdict = {
            "output": url_text,
            "processed": False,
        }

        time_save = time.time()
        with lock:
            self.save_to_file(tmpdict)
            save_delta = time.time() - time_save

            try:
                black_list_url.add(url)
            except:
                pass

            try:
                todo_list_url.remove(url)
            except:
                pass

            time_save_url = time.time()
            self.save_urls()
            save_url_delta = time.time() - time_save_url
            counter += 1

            print(f'Finished processing: Counter: {counter}. Depth: {depth} of {max_depth}. {url}. Save time: {save_delta:.2f}s. Save url time: {save_url_delta:.2f}s. Process time: {process_delta:.2f}s.')
        
    def save_to_file(self, dictionary):
            with open(DATA_PATH, 'rb+') as filehandle:
                filehandle.seek(-1, os.SEEK_END)
                filehandle.truncate()
            
            with open(DATA_PATH, 'a') as f:
                f.seek(0, 2)
                if f.tell() == 0:
                    f.write('[')
                else:
                    f.write(',')
                f.write(json.dumps(dictionary))
                f.write('\n]')
            return

    def save_urls(self):
        json_object = json.dumps(list(black_list_url), indent=4)
        with open(URL_PATH, "w") as outfile:
           outfile.write(json_object)

def load_urls():
    try:
        f = open(URL_PATH)
        black_list_url = set(json.load(f))
        return black_list_url
    except:
        print("No URL file found")  
        return set()
             
urls = {
    'https://poedb.tw/us' : 5,
    'https://poebuilds.cc' : 5,
    'https://poe-vault.com' : 5,
    'https://maxroll.gg/poe' : 5,
    'http://www.vhpg.com': 5,
    'https://poe.ninja' : 3,
    'https://pathofexile.com/forum/view-thread/3409617' : 3,
    'https://old.reddit.com/r/PathOfExileBuilds': 4,
    'https://poewiki.net' : 5
}

async def crawl_thread(c: Crawler, url: str):
    print(f'Started crawling {url}.')
    await c.crawl(url)

crawlers = []
async def main():
    executor = ThreadPoolExecutor(max_workers=10)
    loop = asyncio.get_event_loop()
    tasks = []
    for url, max_depth in urls.items():
        c = Crawler(url, max_depth)
        f = loop.run_in_executor(executor, c.crawl, url)
        tasks.append(f)

    await asyncio.gather(*tasks)
    executor.shutdown()
    return
    
from concurrent.futures import ThreadPoolExecutor
if __name__ == "__main__":
    black_list_url = load_urls()

    asyncio.run(main())

    for c in crawlers:
        c.driver.close()
    

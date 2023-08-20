# https://github.com/Facico/Chinese-Vicuna/tree/master
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
from concurrent.futures import ThreadPoolExecutor

logging.set_verbosity_error()

DATA_PATH = 'data/text_unprocessed2.json'
URL_PATH = 'data/urls_unprocessed2.json'
TODO_PATH = 'data/todo_list.json'

BLACK_LIST_LANG = [ '/cn/', '/kr/', '/ru/', '/jp/', '/tw/', '/po/', '/th/', '/fr/', '/de/', '/es/',  '/br/', '/cn', '/kr', '/ru', '/jp', '/tw', '/th', '/fr', '/de', '/es',  '/br',
                   'recentchanges', '/trade/search', '/edit', '#edit', '/privacy', '/economy', '&diff', '#searchInput', '&oldid', 'userlogin',
                   'facebook', 'instagram', 'youtube', 'twitch', 'discord', 'tiktok', 'linkedin', 'pinterest', 'tumblr', 'vimeo', 'snapchat' ]
alpha = re.compile(r'[a-zA-Z]')
counter = 0

lock = threading.Lock() #threading.Lock()
save_lock = threading.Lock()#threading.Lock()
black_list_url = set()
todo_list_url = {}
closed = 0

DEPTH = 0
MAX_DEPTH = 1
BASE_URL = 2

from bs4 import BeautifulSoup
from textblob import TextBlob

class Crawler:

    def __init__(self, i):
        self.depth = 0
        self.index = i

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1280,1280")
        options.add_argument("--no-sandbox")
        options.add_argument("--enable-javascript")

        self.driver = webdriver.Chrome(options=options)

        self.driver.implicitly_wait(2)
        self.running = True

    def check_url(self):
        with lock:
            if len(todo_list_url) > 0:
                url = list(todo_list_url)[0]
                url_info = todo_list_url.pop(url)
            else:
                url_info = None
                url = None
        return url, url_info

    def stop(self):
        self.running = False
        self.driver.quit()

    def start(self):
        global closed
        self.repeatedAttempts = 0
        try:
            url, url_info = self.check_url()
        except:
            url = None

        while self.running:
            if url != None:
                self.repeatedAttempts = 0
                self.crawl(url, url_info)
            else:
                print(f'Waiting thread: {self.index}')
                time.sleep(1)
                if self.repeatedAttempts > 60:
                    closed += 1
                    print(f'Closed thread: {self.index}. Total closed: {closed}')
                    break
                self.repeatedAttempts += 1
            url, url_info = self.check_url()

    def crawl(self, url: str, depths: list):
        global counter
        base_url = depths[BASE_URL]
        max_depth = depths[MAX_DEPTH]
        depth = depths[DEPTH]
        if depth > max_depth:
            return

        with lock:
            if url in black_list_url or url in todo_list_url or any([lang in url.lower() for lang in BLACK_LIST_LANG]):
                return

        if any(item in url for item in ['pathofexile.com', base_url]):
            pass
        else:
            return

        try:
            self.driver.get(url)
        except:
            return

        html = self.driver.page_source

        if html is None:
            return

        soup = BeautifulSoup(html, 'html.parser')
        a_list = set([urljoin(url, node.get('href')) for node in soup.find_all('a')])

        i = 0
        for a in a_list:
            if a not in black_list_url and any(item in a for item in ['pathofexile.com', base_url]):
                i += 1
                with lock:
                    todo_list_url[a] = [depth + 1, max_depth, base_url]

        soup_extra = BeautifulSoup(html, 'html.parser')
        remove_list = { 'payment', 'buy ', 'preferences', 'partners', 'partner', 'paypal' ,'skrill', 'crypto', 'facebook', 'instagram', 'privacy', 'policy', 'creative', 'commons', 'disclaimers', 'patreon', 'twitter', 'youtube', 'twitch', 'discord', 'reddit', 'tiktok', 'linkedin', 'pinterest', 'tumblr', 'vimeo', 'snapchat', 'whatsapp', 'imprint', 'impressum', 'contact', 'cookies', 'privacy', 'policy', 'legal', 'data', 'protection', 'sitemap', 'accessibility', 'accessable', 'access', 'help', 'faq', 'support', 'donate', 'donation', 'donations', 'report', 'reports', 'advertising', 'cookies', 'cookie'}

        time_save = time.time()
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

        url_text = TextBlob(url_text).string#' '.join([ x for x in b.raw_sentences if ('' if any(y in x.lower() for y in remove_list) else x) ])

        url_text = url_text.replace('\n', ' ')
        url_text = re.sub(r'[^\sa-zA-Z0-9\._-]', '', url_text)
        url_text = re.sub(r'([A-Z])', r' \1', url_text)
        url_text = re.sub('\s+', ' ', url_text)
        url_text = re.sub(r'[^\w\s]', '', url_text).strip()

        url_text = url_text.replace(' Po E ', ' PoE ')
        url_text = url_text.replace(' Po E.', ' PoE.')
        url_text = url_text.replace(' Po E,', ' PoE,')
        url_text = url_text.replace('Po E ', 'PoE ')
        url_text = url_text.replace('P O E', 'PoE')

        process_delta = time.time() - time_save

        if not alpha.match(url_text):
            return

        with lock:
            try:
                black_list_url.add(url)
            except:
                pass

        with save_lock:
            time_save = time.time()
            if url_text != '':
                tmpdict = {
                    "output": url_text,
                    "processed": False,
                }
                self.save_to_file(tmpdict)
            save_delta = time.time() - time_save
            time_save_url = time.time()
            self.save_urls()
            save_url_delta = time.time() - time_save_url
        counter += 1
        print(f'Finished processing: count: {counter}. Depth: {depth} of {max_depth}. Thread {self.index}. {url}. Save time: {save_delta:.2f}s. Save url time: {save_url_delta:.2f}s. Process time: {process_delta:.2f}s.')
        if counter % 100 == 0:
            print(f'Todo list: {len(todo_list_url)}')

    def save_to_file(self, dictionary):

            if not os.path.exists(DATA_PATH):
                with open(DATA_PATH, 'w') as f:
                    f.write('[]')

            with open(DATA_PATH, 'rb+') as filehandle:
                filehandle.seek(-1, os.SEEK_END)
                filehandle.truncate()

            with open(DATA_PATH, 'a') as f:
                f.seek(0, 2)
                #if f.tell() == 0:
                #    f.write('[')
                #else:
                f.write(',')
                f.write(json.dumps(dictionary))
                f.write('\n]')
            return

    def save_urls(self):
        if not os.path.exists(URL_PATH):
            with open(URL_PATH, 'w') as f:
                f.write('[]')

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

base_urls = {
    'poedb.tw/us' : 6,
    'poebuilds.cc' : 5,
    'poe-vault.com' : 5,
    'maxroll.gg/poe' : 5,
    'poe.ninja' : 4,
    'pathofexile.com/forum/view-thread/3409617' : 4,
    'old.reddit.com/r/PathOfExileBuilds': 4,
    'poewiki.net' : 6
}

def save_todo_list():
    print('Saving todo list...')
    try:
        print(len(todo_list_url))
        json_object = json.dumps(todo_list_url, indent=4)
        with open(TODO_PATH, "w") as outfile:
            outfile.write(json_object)
    except:
        print('Error saving todo list')


async def main():

    crawlers = []

    num_threads = 16
    executor = ThreadPoolExecutor(max_workers=num_threads)
    loop = asyncio.get_event_loop()

    https = 'https://'

    for url in base_urls:
        if url not in black_list_url:
            todo_list_url[https + url] = [0, base_urls[url], url]
    tasks = []

    try:
        for i in range(num_threads):
            c = Crawler(i)
            crawlers.append(c)
            f = loop.run_in_executor(executor, c.start)
            tasks.append(f)

        print('Starting')
        await asyncio.gather(*tasks)
        for c in crawlers:
            c.stop()
    except KeyboardInterrupt:
        for c in crawlers:
            c.stop()

        for task in tasks:
            task.cancel()

        save_todo_list()

    executor.shutdown()
    print('Finished')

    return

def get_todo_list():
    try:
        f = open(TODO_PATH)
        todo_list_url = json.load(f)
        return todo_list_url
    except:
        print("No TODO file found")
        return {}

todo_list_url = get_todo_list()
print(f'Todo list: {len(todo_list_url)}')

if __name__ == "__main__":

    black_list_url = load_urls()
    counter += len(black_list_url)

    asyncio.run(main())

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
import requests


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

TEMPERATURE = 0.05
MAX_NEW_TOKENS = 1024
API_HOST = 'http://localhost'
API_PORT = 5000
CTX_MAX = 16384

from bs4 import BeautifulSoup
from textblob import TextBlob

# https://github.com/flurb18/babyagi4all-api
def ooba_call(prompt: str):
    URI=f'{API_HOST}:{API_PORT}/v1/completions'

    request = {
        'prompt': prompt[:CTX_MAX],
        'max_new_tokens': MAX_NEW_TOKENS,
        'do_sample': True,
        'temperature': TEMPERATURE,
        'top_p': 0.1,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': [],
        "max_tokens": MAX_NEW_TOKENS
    }

    response = requests.post(URI, json=request)

    if response.status_code == 200:
        j = response.json()
        return j['choices'][0]['text']
    else:
        print("Something went wrong accessing api")

class Crawler:

    def __init__(self, i, lock):
        self.depth = 0
        self.index = i
        self.lock = lock

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1280,1280")
        options.add_argument("--no-sandbox")
        options.add_argument("--enable-javascript")

        self.driver = webdriver.Chrome(options=options)

        self.driver.implicitly_wait(2)
        self.running = True

        self.remove_list = { 'payment', 'buy ', 'preferences', 'partners', 'partner', 'paypal' ,'skrill', 'crypto', 'facebook', 'instagram', 'privacy', 'policy', 'creative', 'commons', 'disclaimers', 'patreon', 'twitter', 'youtube', 'twitch', 'discord', 'reddit', 'tiktok', 'linkedin', 'pinterest', 'tumblr', 'vimeo', 'snapchat', 'whatsapp', 'imprint', 'impressum', 'contact', 'cookies', 'privacy', 'policy', 'legal', 'data', 'protection', 'sitemap', 'accessibility', 'accessable', 'access', 'help', 'faq', 'support', 'donate', 'donation', 'donations', 'report', 'reports', 'advertising', 'cookies', 'cookie', 'network policy', 'privacy policy', 'request', 'browsing', 'script', 'settings', 'IP address', 'IP', 'address', 'ticket', 'applications', 'string', 'agent', 'network', 'Your request has been blocked', 'halted', 'restriction', 'establish', 'account'}

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


    def summarize(self, text: str) -> str:
        #with self.lock:
        return ooba_call(f'Your task is to rewrite using the original words and wordings. Generate a concise rewrite of {text} without losing important information, and capture the main points and key details. Start by mention what the text is about. Do not mention this rewrite. Only write using the original texts wording. Only use the text provided. Remove all mentions of {self.remove_list} if they exist. ')

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
            if any(x in tag.get_text().lower() for x in self.remove_list):
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


            time_save = time.time()

            if url_text != '':
                url_text = self.summarize(url_text)
                if url_text != ' ' and url_text != '':

                    tmpdict = {
                        "output": url_text,
                        "processed": False,
                    }
                    self.save_to_file(tmpdict)
                with save_lock:
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

    num_threads = 8
    executor = ThreadPoolExecutor(max_workers=num_threads)
    loop = asyncio.get_event_loop()

    https = 'https://'

    for url in base_urls:
        if url not in black_list_url:
            todo_list_url[https + url] = [0, base_urls[url], url]
    tasks = []

    try:
        for i in range(num_threads):
            c = Crawler(i, lock)
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

# https://github.com/Facico/Chinese-Vicuna/tree/master
import requests
import os
from bs4 import BeautifulSoup, SoupStrainer
import re
#from trafilatura import fetch_url, extract
#from trafilatura.utils import sanitize, trim
from urllib.parse import urljoin
import json
from torch import cuda, bfloat16
import time
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers import pipeline, AutoTokenizer, AutoModel, LlamaForCausalLM, BitsAndBytesConfig, BartForCausalLM
from transformers.utils import logging
import asyncio
import random

import concurrent.futures

logging.set_verbosity_error()

MODEL_PATH = 'lmsys/vicuna-7b-v1.5-16k'
DATA_PATH = 'data/text.json'
URL_PATH = 'data/urls.json'
MODEL_PATH = 'kabita-choudhary/finetuned-bart-for-conversation-summary' #works well and is small

BLACK_LIST_LANG = ['/cn/', '/kr/', '/ru/', '/jp/', '/tw/', '/po/', '/th/', '/fr/', '/de/', '/es/']
alpha = re.compile(r'[a-zA-Z]')
counter = 0

MAXLEN = 1024#4096
lock = threading.Lock()
black_list_url = set()
todo_list_url = set()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device_map = "auto"
num_models = 4
#generator = pipeline("summarization", tokenizer=tokenizer, model=MODEL_PATH, device_map=device_map)

pipelines = [pipeline("summarization", tokenizer=tokenizer, model=MODEL_PATH, device_map=device_map) for _ in range(num_models)]

class Crawler:

    def __init__(self, url: str, max_depth: int):

        self.url = url
        self.base_url = url
        self.max_depth = max_depth
        self.depth = 0

        options = Options()
        options.headless = True
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(1)



    def generate_instruction(self, url_text):
        generator = random.choice(pipelines)
        if len(url_text) > MAXLEN:
            chunks = len(url_text)//MAXLEN
            url_text_chunks = [ url_text[i:i+MAXLEN] for i in range(0, MAXLEN*chunks+1, MAXLEN) ]
            tmp_MAXLEN = MAXLEN//(len(url_text_chunks))
            if tmp_MAXLEN < 6:
                tmp_MAXLEN = 6
            url_text_chunks = generator(url_text_chunks, min_length=5, max_length=tmp_MAXLEN)
            url_text_chunks = " ".join([ chunk['summary_text'] for chunk in url_text_chunks ])
            output_instruction = url_text_chunks[:MAXLEN]
        else:
            output_instruction = generator(url_text, max_length=MAXLEN)
            #self.generator(prompt, max_length=MAXLEN, return_full_text=False)#(f'Instruction based on "${url_text}", a Path of Exile I recommend is ', max_length=MAXLEN, return_full_text=False)
            output_instruction = output_instruction[0]['summary_text']#['generated_text']
        output_instruction = f'Provide more information about the following in Path of Exile: {output_instruction}'
        return output_instruction

    def crawl(self, url: str, depth=0):
        global counter
        max_depth = self.max_depth
        base_url = self.base_url

        if depth > max_depth:
            return

        with lock:
            if url in black_list_url or url in todo_list_url or any([lang in url for lang in BLACK_LIST_LANG]):
                return

        if base_url not in url:
            return

        self.driver.get(url)
        html = self.driver.page_source

        if html is None:
            return
        #print(url)

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

        url_text = soup.get_text(strip=True)

        if url_text is None:
            return

        url_text = url_text.replace('\n', ' ')
        url_text = re.sub(r'[^\sa-zA-Z0-9\._-]', '', url_text)
        url_text = re.sub(r'([A-Z])', r' \1', url_text)
        url_text = re.sub('\s+', ' ', url_text)
        url_text = re.sub(r'[^\w\s]', '', url_text).strip()

        if not alpha.match(url_text):
            return

        start_generate = time.time()
        output_instruction = self.generate_instruction(url_text)
        denerate_delta = time.time() - start_generate

        tmpdict = {
            "instruction": output_instruction,
            "input": "",
            "output": url_text
        }

        time_save = time.time()
        with lock:
            self.save_to_file(tmpdict)
            save_delta = time.time() - time_save

            black_list_url.add(url)
            todo_list_url.remove(url)

            time_save_url = time.time()
            self.save_urls()
            save_url_delta = time.time() - time_save_url
            counter += 1

            print(f'Finished processing: Counter: {counter}. Depth: {depth} of {max_depth}. {url}. Time: {denerate_delta:.2f}s. Save time: {save_delta:.2f}s. Save url time: {save_url_delta:.2f}s.')

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
    'https://poedb.tw/us' : 3,
    'https://poebuilds.cc' : 3,
    'https://poe-vault.com' : 3,
    'https://maxroll.gg/poe' : 3,
    'https://poe.ninja' : 2,
    'https://pathofexile.com/forum/view-thread/3409617' : 1,
    'https://old.reddit.com/r/PathOfExileBuilds': 2,
    'https://poewiki.net' : 3
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
    tasks = []
    for url, max_depth in urls.items():
        c = Crawler(url, max_depth)
        crawlers.append(c)
        await loop.run_in_executor(executor, crawl_thread)
        #task = loop.create_task(crawl_thread(c, url))
        #tasks.append(task)
    await asyncio.gather(*tasks)

from concurrent.futures import ThreadPoolExecutor
if __name__ == "__main__":
    black_list_url = load_urls()

    asyncio.run(main())

    for c in crawlers:
        c.driver.close()

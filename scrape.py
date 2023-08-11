# https://github.com/Facico/Chinese-Vicuna/tree/master
import requests
import os
from bs4 import BeautifulSoup, SoupStrainer
import re
from trafilatura import fetch_url, extract
from trafilatura.utils import sanitize, trim
from urllib.parse import urljoin
import json
from torch import cuda, bfloat16

from transformers import pipeline, AutoTokenizer, AutoModel, LlamaForCausalLM, BitsAndBytesConfig, BartForCausalLM
from transformers.utils import logging

logging.set_verbosity_error()

MODEL_PATH = 'lmsys/vicuna-7b-v1.5-16k'
DATA_PATH = 'data/text.json'
URL_PATH = 'data/urls.json'
MODEL_PATH = 'kabita-choudhary/finetuned-bart-for-conversation-summary' #works well and is small

MAXLEN = 1024#4096

class Crawler:

    def __init__(self):
        self.todo_list_url = set()
        self.black_list_url = set()

        self.counter = 0
        self.dictionary_list = []
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        device_map = "auto"
        USE_8bit = True
        self.model = BartForCausalLM.from_pretrained(
            MODEL_PATH,
            load_in_8bit=USE_8bit,
            device_map=device_map,
            quantization_config=bnb_config
        )

        self.generator = pipeline("summarization", tokenizer=self.tokenizer, model=MODEL_PATH, device_map=device_map)#, tokenizer=self.tokenizer, task="text-generation")


    def generate_instruction(self, url_text):
        """"""
        prompt = (
            f"""Below is a task that is related to Path of Exile. Write a clear response that appropriately describe an instruction for the task, such as proving more information about the task.

### Task:
{url_text}


### Response instruction:
"""
    )
        if len(url_text) > MAXLEN:
            chunks = len(url_text)//MAXLEN
            url_text_chunks = [ url_text[i:i+MAXLEN] for i in range(0, MAXLEN*chunks+1, MAXLEN) ]
            tmp_MAXLEN = MAXLEN//(len(url_text_chunks))
            if tmp_MAXLEN < 6:
                tmp_MAXLEN = 6
            url_text_chunks = " ".join([ self.generator(chunk, min_length=5, max_length=tmp_MAXLEN)[0]['summary_text'] if len(chunk) > 5 else chunk for chunk in url_text_chunks ])
            output_instruction = url_text_chunks[:MAXLEN]
        else:
            output_instruction = self.generator(url_text, max_length=MAXLEN)#self.generator(prompt, max_length=MAXLEN, return_full_text=False)#(f'Instruction based on "${url_text}", a Path of Exile I recommend is ', max_length=MAXLEN, return_full_text=False)
            output_instruction = output_instruction[0]['summary_text']#['generated_text']
        output_instruction = f'Provide more information about the following in Path of Exile: {output_instruction}'
        return output_instruction

    def crawl(self, url: str, base_url: str, depth=0, max_depth=3):
        if url in self.black_list_url or url in self.todo_list_url:
            return

        self.todo_list_url.add(url)
        self.counter += 1
        print(f'Counter: {self.counter}. Processing: {url}. Depth: {depth} of {max_depth}')

        if base_url not in url:
            return

        html = fetch_url(url)
        if html is None:
            return

        # get content on the page
        url_text = extract(html)
        url_text = sanitize(url_text)
        url_text = trim(url_text)

        if url_text is None:
            return

        url_text = re.sub(r'[^\sa-zA-Z0-9\._-]', '', url_text)
        url_text_instruction = re.sub(' +', ' ', url_text)
        url_text_instruction = url_text

        #if len(url_text) >= MAXLEN-512:
        #    url_text_instruction = url_text[:MAXLEN-512]

        output_instruction = self.generate_instruction(url_text_instruction)
        tmpdict = {
            "instruction": output_instruction,
            "input": "",
            "output": url_text
        }

        self.save_to_file(tmpdict)

        if depth >= max_depth:
            return

        # get links on the page
        to_visit = []
        soup = BeautifulSoup(html, parse_only=SoupStrainer('a'))
        a_list = set([urljoin(url, node.get('href')) for node in soup.find_all('a')])
        for a in a_list:
            if a not in to_visit and a not in self.black_list_url:
                to_visit.append(a)
                self.crawl(a, base_url, depth=depth+1)

        self.black_list_url.add(url)
        self.save_urls()

    def save_to_file(self, dictionary_list):
        data = []
        try:
            f = open(DATA_PATH)
            data = json.load(f)
        except:
            pass
        data.append(dictionary_list)
        json_object = json.dumps(data, indent=4)
        with open(DATA_PATH, "w") as outfile:
            outfile.write(json_object)

    def load_urls(self):
        try:
            f = open(URL_PATH)
            self.black_list_url = set(json.load(f))
        except:
            print("No URL file found")

    def save_urls(self):
        json_object = json.dumps(list(self.black_list_url), indent=4)
        with open(URL_PATH, "w") as outfile:
            outfile.write(json_object)

urls = {
    'https://poedb.tw/us' : 5,
    'https://www.poebuilds.cc' : 4,
    'https://www.poe-vault.com' : 4,
    'https://maxroll.gg/poe/category/getting-started' : 2,
    'https://poe.ninja/builds/challenge' : 2,
    'https://www.pathofexile.com/forum/view-thread/3409617' : 1,
    'https://old.reddit.com/r/PathOfExileBuilds': 2
}

if __name__ == "__main__":
    c = Crawler()
    c.load_urls()

    for url in urls:
        base_url = os.path.dirname(url)
        c.crawl(url, base_url, max_depth=urls[url])
    pass

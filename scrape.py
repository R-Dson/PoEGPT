# https://github.com/Facico/Chinese-Vicuna/tree/master
import requests
from bs4 import BeautifulSoup, SoupStrainer
import re
from trafilatura import fetch_url, extract
from trafilatura.utils import sanitize, trim
from urllib.parse import urljoin
import json
from torch import cuda, bfloat16
#import finetune

MAXLEN = 4096
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, LlamaForCausalLM, BitsAndBytesConfig

MODEL_PATH = 'lmsys/vicuna-7b-v1.5-16k'
DATA_PATH = 'data/text.json'
class Crawler:
    
    def __init__(self):
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
        self.model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH,
            load_in_8bit=USE_8bit,
            device_map=device_map,
            quantization_config=bnb_config
        )

        self.generator = pipeline(model=self.model, tokenizer=self.tokenizer, task="text-generation")

    def crawl(self, url: str, depth=0, max_depth=3):
        self.black_list_url.add(url)
        self.counter += 1

        html = fetch_url(url)

        # get content on the page
        url_text = extract(html)
        url_text = sanitize(url_text)
        url_text = trim(url_text)
        url_text = re.sub(r'[^\sa-zA-Z0-9\._-]', '', url_text)
        url_text_instruction = re.sub(' +', ' ', url_text)
        url_text_instruction = url_text
        if len(url_text) >= MAXLEN-512:
            url_text_instruction = url_text[:MAXLEN-512]
        l = len(url_text_instruction)
        l2 = len(url_text)

        output_instruction = self.generator(f'Instruction based on "${url_text_instruction}", a Path of Exile build I recommend is ', max_length=MAXLEN, return_full_text=False)
        output_instruction = output_instruction[0]['generated_text']
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
                self.crawl(a, depth=depth+1)
        
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
                
urls = {
    'https://poedb.tw/us' : 3,
    'https://www.poebuilds.cc' : 4,
    'https://www.poe-vault.com' : 4,
    'https://maxroll.gg/poe/category/getting-started' : 2,
    'https://poe.ninja/builds/challenge' : 2,
    'https://www.pathofexile.com/forum/view-thread/3409617' : 1
}

if __name__ == "__main__":
    c = Crawler()
    for url in urls:
        c.crawl(url, max_depth=urls[url])
    pass

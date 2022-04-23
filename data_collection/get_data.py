MAX_REPO_PER_LANG = 5000
LANG = ['Python']

from bs4 import BeautifulSoup
import requests
from tqdm import *
import time
import json
import re
import os
import logging
from github import Github, UnknownObjectException, RateLimitExceededException
from hn import search_by_date
from markdown import markdown
from langdetect import detect
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='en')
translate_wrapper = lambda x: translator.translate(x) 
# using an access token
g = 'Your Github Token'

def jprint(content):
    print(json.dumps(content, indent=2))

def divide(text, chunk_len=2048):
    n_chunks = len(text) // chunk_len
    return [
        text[i*chunk_len: i*chunk_len+chunk_len] if i != n_chunks - 1 else text[i*chunk_len::] for i in range(n_chunks)
    ]

def chunk_translate_en(text: str):
    return "".join(
        list(
            map(
                translate_wrapper, divide(text)
            )
        )
    )

def get_hn_comments(topic: str) -> str:
    '''
    Arguments: 
        - topic (str) - form of f'{author_name}/{repo_name}' works best.
    Returns:
        str - concatenated comments
    '''
    text = ''
    for index, r in enumerate(search_by_date(q=topic, stories=True, num_comments__gt=0)):
        if index >= 5:
            break
        hn_comments_raw = requests.get(f'http://hn.algolia.com/api/v1/items/{r["objectID"]}').json()['children']
        hn_comments_text = '<HN_SEP>'.join(
            [
                BeautifulSoup(x['text']).text for x in hn_comments_raw if x['text'] is not None and len(x['text']) > 0
            ]
        )
        text += f"{hn_comments_text}<HN_SEP>"
    return text

if __name__ == '__main__':
    logging.basicConfig(filename='magi_data.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    results = []
    # for lang in LANG:
    lang = 'Python'
    repositories = g.search_repositories(query=f'stars:>100 language:{lang}')
        # results[lang] = []
    success = 0
    for index, repo in enumerate(repositories):
        while True:
            try:
                logging.info(f'{lang}: {success:3d}/{MAX_REPO_PER_LANG}')
                # print(f'{lang}: {success:3d}/500' , end='\r')
                if success >= MAX_REPO_PER_LANG:
                    logging.info(f'{lang}: {success:3d}/{MAX_REPO_PER_LANG}, finished')
                    break
                root_file_list = repo.get_contents("")
                readme_filename = None
                for c in root_file_list:
                    if "README" in c.name:
                        readme_filename = c.name
                if readme_filename is None:
                    continue
                if repo.get_contents(readme_filename) is None:
                    continue
                if type(repo.get_contents(readme_filename)) is list:
                    dl_url = repo.get_contents(readme_filename)[0].download_url
                else:
                    dl_url = repo.get_contents(readme_filename).download_url
                # readme = ''.join(
                #             BeautifulSoup(
                #                 markdown(
                #                     requests.get(dl_url).text
                #                 )
                #             ).findAll(text=True)
                #         )
                readme = requests.get(dl_url).text
                readme_lang = detect(readme[0:(512 if len(readme) <= 512 else -1)])
                if not readme_lang == 'en':
                    while True:
                        try:
                            readme = chunk_translate_en(readme)
                        except Exception as e:
                            logging.warning(f"{e}")
                        finally: 
                            break
                while True:
                    try:
                        hn_comments = get_hn_comments(repo.full_name)
                    except Exception as e:
                        logging.warning(f"{e}")
                    finally: 
                        break
                repo_info = {
                    'name': repo.full_name,
                    'link': repo.html_url,
                    'tags': repo.get_topics(),
                    'stars': repo.stargazers_count,
                    'description': repo.description,
                    'readme': readme,
                    'orig_lang': readme_lang,
                    'hn_comments': hn_comments
                }
                results.append(repo_info)
                success += 1
            except RateLimitExceededException as e:
                time.sleep(120)
            finally:
                break
        if (index + 1) % 500 == 1:
            with open('ghv6.json', 'w') as f:
                json.dump(results, f, indent=2)
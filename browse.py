import streamlit as st
import json
import re
from pathlib import Path
from fuzzywuzzy import process

emoj = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        "]+", flags = re.UNICODE)

def remove_emojis(data):
    # https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    return re.sub(emoj, '', data)

def get_all_json(curr_path: Path, suffix='json'):
    files = list(curr_path.glob(f'*.{suffix}'))
    for child in curr_path.iterdir():
        if not child.is_dir():
            continue
        files += get_all_json(child, suffix=suffix)
    return files

files = get_all_json(Path('./'))
selected_file = st.selectbox('Select a file to read', files)
st.session_state['file'] = selected_file

with open(st.session_state['file'], 'r') as f:
    repo_dict = json.load(f)

all_langs = set([x['lang'] for x in repo_dict])

repo_search = {
    remove_emojis(f'{r["name"]}, {r["description"]}'): r for r in repo_dict
}

search_text = st.text_input('Search a repository...', value='')

search_scope = st.multiselect(
    'Language', all_langs, all_langs
)

if st.button('Apply'):
    result = process.extract(search_text, list(repo_search.keys()), limit=10)
    
    for (r, _) in result:
        if not (repo_search[r]['lang'] in search_scope):
            continue
        st.markdown(f"🗂  [{repo_search[r]['name']}]({repo_search[r]['link']})")
        st.markdown(f"⭐️  {repo_search[r]['stars']} | {repo_search[r]['description']}")
        st.markdown(f'💾  Data:')
        st.json(repo_search[r], expanded=False)
        st.markdown('''<hr style="height:2px;border:none;color:#CCC;background-color:#CCC;" />''', unsafe_allow_html=True)
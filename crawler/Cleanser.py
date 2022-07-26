import json
import string
from tqdm import tqdm

list = []
data = ""

print("\t[open...]")
with open("../data/list//musicList.json", 'r', encoding="utf-8") as f:
    data = json.load(f)
print("\t[open complete]")

def cleansing(m):
    m["artists"] = m["artists"].replace("<", "(")
    m["artists"] = m["artists"].replace(">", ")")

    m["MusicName"] = m["MusicName"].replace("<", "(")
    m["MusicName"] = m["MusicName"].replace(">", ")")

    m["lyrics"] = m["lyrics"].translate(str.maketrans('', '', string.punctuation))
    m["lyrics"] = m["lyrics"].replace(" ", "")
    m["lyrics"] = m["lyrics"].replace("﻿", "")
    m["lyrics"] = m["lyrics"].replace("​", "")
    m["lyrics"] = m["lyrics"].replace("", "")
    return m

end = len(data)
print("\t[cleansing...]")
for idx in tqdm(range(0, end)):
    element = cleansing(data[idx])
    if element :
        list.append(element)

print("\t[cleansing complete]")

print("\t[save...]")
with open("../data/list//musicList.json", 'w', encoding="utf-8") as f:
    json.dump(list, f, ensure_ascii=False)

print("\t[save complete]")
import json

import numpy as np
from tqdm import tqdm
from sen2vec import sen2vec

def getTokenizedData():
    print("[Data open]")
    print("\ttokenized_data open...")
    with open("../../data/token/tokenized_data.json", 'r', encoding='utf-8') as f:
        tokenized_data = json.loads(f.read())
        del f
    print("\tcomplete")

    return tokenized_data

tokenData = getTokenizedData()

print("[convert vector]")
vecData = []
for lyrics in tqdm(tokenData):
    lyricsVec = []
    for sentence in lyrics:
        lyricsVec.append(sen2vec(sentence).tolist())

    vecData.append(lyricsVec)

print("[File save]")
print("\tembedded lyrics file save...")
with open("../../data/lyricsVector/lyricsVector.json", 'w') as f:
    json.dump(vecData, f)
print("\tcomplete")
import json
import numpy as np
from tqdm import tqdm
from sent2vec import Sent2vec
from gensim import utils


s2v = Sent2vec("../../data/model/music_w2v_100_5_sg.model")

def getTokenizedData(idx):
    with open("../../data/token/parts/" + str(idx) + ".json", 'r', encoding='utf-8') as f:
        tokenized_data = json.loads(f.read())
        del f

    return tokenized_data



print("[convert vector]")

vecData = []
totalVec = 0
for idx in tqdm(range(0,129)):
    tokenData = getTokenizedData(idx)

    for lyrics in tokenData:
        lyricsVec = []
        for sentence in lyrics:
            lyricsVec.append(s2v.sent2vec(sentence))
            totalVec += 1
        vecData.append(lyricsVec)

    del tokenData



print("[create model]")

with utils.open("../../data/sentVecData/music_s2v_5_sg_avg_temp.file", 'wb') as f:
    print("\tembedded lyrics file save...")
    f.write(f"{totalVec} {s2v.getDim()}\n".encode('utf8'))

    musicIdx = 0
    for lyrics in tqdm(vecData):
        lyricsIdx = 0

        for sentence in lyrics:
            f.write(f"{musicIdx},{lyricsIdx} {' '.join(repr(val) for val in sentence)}\n".encode('utf8'))
            lyricsIdx += 1
        musicIdx += 1

    f.close()
    print("\tcomplete")
import json
import numpy as np
from tqdm import tqdm
from sent2vec import Sent2vec as S2V
from gensim import utils

file_count = 129
file_path = "../../data/sentVecData/lyrics_vector_data_300000_avgFlow.file"
s2v = S2V("../../data/model/music_w2v_100_5_sg.model")


def getTokenizedData(idx):
    with open("../../data/token/parts/" + str(idx) + ".json", 'r', encoding='utf-8') as f:
        tokenized_data = json.loads(f.read())
        del f

    return tokenized_data



print("[convert vector]")

vecData = []
totalVec = 0
for idx in tqdm(range(0,file_count)):
    tokenData = getTokenizedData(idx)

    for lyrics in tokenData:
        lyricsVec = []
        for sentence in lyrics:
            lyricsVec.append(s2v.sent2vec(sentence))
            totalVec += 1
        vecData.append(lyricsVec)

    del tokenData



print("[create model]")

with utils.open(file_path, 'wb') as f:
    print("\tembedded lyrics file save...")
    f.write(f"{totalVec} {s2v.get_dim()}\n".encode('utf8'))

    musicIdx = 0
    for lyrics in tqdm(vecData):
        lyricsIdx = 0

        for sentence in lyrics:
            f.write(f"{musicIdx},{lyricsIdx} {' '.join(repr(val) for val in sentence)}\n".encode('utf8'))
            lyricsIdx += 1
        musicIdx += 1

    f.close()
    print("\tcomplete")
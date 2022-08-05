import json
import pandas as pd
import numpy
from gensim.models import KeyedVectors
import numpy as np
from Sent2vec import sent2vec, getDim
from konlpy.tag import Okt

print("[sen2vec]")
print("\tload model...")
model = KeyedVectors.load_word2vec_format("../../data/s2vModel/music_s2v_5_sg_avg.model") #C:\Users\백대환\Desktop\IdeaProjects\SIG_2022하계\data\model\music_w2v
print("\tcomplete")

embedding_dim = model.vectors.shape[1]
ZERO_VEC = np.zeros(embedding_dim)

sentence = ""
okt = Okt()

simVecList = []

def getDataFrame(path) :
    print("\topen dataFrame...")
    with open(path, 'r') as f:
        data = json.loads(f.read())
        del f
    print("\topen complete")

    print("\tmake table...")
    df = pd.DataFrame(data)
    del data

    return df


df = getDataFrame("../../data/dataFrame/musicDataFrame.json")

while sentence != "0":
    sentence = input("Enter sentence : ")

    tokenized_sentence = okt.morphs(sentence, stem=True)
    print(tokenized_sentence)

    vec = sent2vec(tokenized_sentence)
    simVecList = model.most_similar(vec)

    for simVec in simVecList:
        v = simVec[0].split(",")
        musicNum = v[0]
        sentNum = int(v[1])

        music = df.loc[musicNum]

        musicName = music["musicName"]
        artists = music["artists"]
        simSent = music["lyrics"][sentNum]

        print(f"music \t\t: {musicName}")
        print(f"artists \t: {artists}")
        print(f"비슷한 가사 \t: {simSent}\n")
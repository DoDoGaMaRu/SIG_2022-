import json
import pandas as pd
from vectorSearcher import VectorSearcher
from bin.sentenceEmbedding.sent2vec import Sent2vec as S2V
from konlpy.tag import Okt



def loadDataFrame(path) :
    print("\topen dataFrame...")
    with open(path, 'r') as f:
        data = json.loads(f.read())
        del f
    print("\topen complete")

    print("\tmake table...")
    df = pd.DataFrame(data)
    del data

    return df

print("[Sent2vec]")
s2v = S2V("../../data/model/music_w2v_100_5_sg.model")

print("[sentence vector data]")
print("\tload data...")
lyricsVS = VectorSearcher.load_word2vec_format("../../data/sentVecData/lyrics_vector_data_300000_avg.file") #C:\Users\백대환\Desktop\IdeaProjects\SIG_2022하계\data\model\music_w2v
print("\tcomplete")

print("[dataFrame]")
df = loadDataFrame("../../data/dataFrame/musicDataFrame_300000.json")
okt = Okt()
sentence = ""

while sentence != "0":
    sentence = input("Enter sentence : ")

    tokenized_sentence = okt.morphs(sentence, stem=True)
    print(tokenized_sentence)

    vec = s2v.sent2vec(tokenized_sentence)
    simVecList = lyricsVS.most_similar(vec)

    for simVec in simVecList:
        musicIdx, sentIdx = [int(x) for x in simVec[0].split(",")]

        music = df.loc[str(musicIdx)]

        musicName = music["musicName"]
        artists = music["artists"]
        simSent = music["lyrics"][sentIdx]

        print(f"music \t\t: {musicName}")
        print(f"artists \t: {artists}")
        print(f"비슷한 가사 \t: {simSent}\n")
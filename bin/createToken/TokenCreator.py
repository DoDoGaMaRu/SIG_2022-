import json
import pandas as pd
import multiprocessing
from konlpy.tag import Okt
from tqdm import tqdm


def createTokenJson(df, fileName) :
    okt = Okt()

    tokenized_data = []
    for lyrics in tqdm(df['lyrics']):
        tokenized_music = []
        for sentence in lyrics:
            tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
            tokenized_music.append(tokenized_sentence)

            del tokenized_sentence

        tokenized_data.append(tokenized_music)


    with open("../../data/token/parts/" + fileName + ".json", 'w', encoding="utf-8") as f:
        json.dump(tokenized_data, f, ensure_ascii=False)

    del df
    del tokenized_data



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



def run(count, thread, size) :
    df = getDataFrame("../../data/dataFrame/musicDataFrame.json")

    print("\trun processes...")
    processes = []
    for idx in range(thread * count, thread * (count + 1)) :
        start = size * idx
        end = size * (idx + 1)
        p = multiprocessing.Process(target=createTokenJson, args=[df.iloc[start:end], str(idx)])
        p.start()               # process 시작
        processes.append(p)

    del df

    for process in processes:
        process.join()



def makeFile(thread, loop) :
    token_full = []

    print("\tmerge token files...")
    for idx in tqdm(range(0, thread * loop + 1)) :
        with open("../../data/token/parts/" + str(idx) + ".json", 'r', encoding="utf-8") as f:
            data = json.loads(f.read())

            token_full += data
            del data
            del f

    print("\tsave tokenized_data...")
    with open("../../data/token/tokenized_data.json", 'w', encoding="utf-8") as f:
        json.dump(token_full, f, ensure_ascii=False)


# main
if __name__ == '__main__':
    thread = 16
    loop = 10

    print("[check size...]")
    df = getDataFrame("../../data/dataFrame/musicDataFrame.json")
    full_size = len(df)
    size = full_size // thread // loop
    print("\tfull_size = " + str(full_size))

    createTokenJson(df.iloc[thread * loop * size:], str(thread * loop))
    del df

    for count in range(0, loop) :
        print("[Run " + str(count) + "]")
        run(count, thread, size)


    print("[make file...]")
    makeFile(thread, loop)
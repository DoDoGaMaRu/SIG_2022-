import json
import pandas as pd
import multiprocessing
from konlpy.tag import Okt
from tqdm import tqdm


def createTokenJson(df, fileName) :
    okt = Okt()

    tokenized_data = []
    for lyrics in tqdm(df['lyrics']) :
        for sentence in lyrics :
            tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화

            if tokenized_sentence:
                tokenized_data.append(tokenized_sentence)

            del tokenized_sentence

    with open("../data/token//" + fileName + ".json", 'w', encoding="utf-8") as f:
        json.dump(tokenized_data, f, ensure_ascii=False)

    del tokenized_data



def getDataFrame(path) :
    print("\topen dataFrame...")
    with open(path, 'r') as f:
        data = json.loads(f.read())
    print("\topen complete")

    print("\tmake table...")
    df = pd.DataFrame(data)
    del data

    return df



def run(count, thread, size) :
    df = getDataFrame("../data/dataFrame//musicDataFrame.json")

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

    for idx in range(0, thread * loop) :
        with open("../data/token/" + str(idx) + ".json", 'r', encoding="utf-8") as f:
            data = json.loads(f.read())

            token_full += data
            del data

    with open("../data/token//token_full.json", 'w', encoding="utf-8") as f:
        json.dump(token_full, f, ensure_ascii=False)


# main
if __name__ == '__main__':
    thread = 8
    loop = 4

    print("[check size...]")
    df = getDataFrame("../data/dataFrame//musicDataFrame.json")
    size = len(df) // thread // loop
    del df

    for count in range(0,loop) :
        print("[Run " + str(count) + "]")
        run(count, thread, size)

    print("[make file...]")
    makeFile(thread, loop)
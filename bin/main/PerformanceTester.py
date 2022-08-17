from musicFinder import MusicFinder
from tqdm import tqdm
import random
import time

df_path = "../../data/dataFrame/musicDataFrame_short.json"
w2v_model_path = "../../data/model/music_w2v_100_5_sg.model"
lyrics_vec_path = "../../data/sentVecData/lyrics_vector_data_short_avg.file"


mf = MusicFinder(df_path=df_path, w2v_model_path=w2v_model_path, lyrics_vec_path=lyrics_vec_path)
df = mf.df
w2v_kv = mf.s2v.w2v_kv

def get_random_sent(df_idx):
    music = df.loc[str(df_idx)].copy()
    lyrics = music["lyrics"]

    min_count = 5
    del_word_count = 2
    change_word_count = 2
    result = None

    for sent in lyrics:                 # 5개 단어 이상의 문장 찾기
        split_sent = sent.split()
        count = len(split_sent)
        if min_count <= count:
            result = split_sent
            break

    if result != None:
        if random.choice([True, False]):    # n개의 단어 삭제
            for count in range(0, del_word_count):
                del_idx = random.randrange(0,len(result))
                del result[del_idx]
        else:                               # n개 단어 다른 단어로 대체
            for count in range(0, change_word_count):
                change_idx = random.randrange(0,len(result))
                try:
                    sim_word = w2v_kv.most_similar(negative=result[change_idx], topn=1)[0][0]
                    result[change_idx] = sim_word
                except KeyError:
                    del result[change_idx]

        result = ' '.join(result)

    return result



print("[Run performance test]")
num_of_trials = 100

for x in range(0, 5):
    sum_time = 0
    trials_count = num_of_trials
    success_count = 0
    for count in tqdm(range(0, num_of_trials)):
        df_idx = random.randrange(0,len(df))
        sentence = get_random_sent(df_idx)
        if sentence == None:
            trials_count -= 1
            continue

        music = df.loc[str(df_idx)]
        music_num = music["musicNum"]

        start = time.time()
        sim_music_list = mf.find_music(sentence, topn=10)
        sum_time += (time.time() - start)
        sim_music_num_list = [sim_music["musicNum"] for sim_music in sim_music_list]

        if music_num in sim_music_num_list:
            success_count += 1

    success_rate = success_count / trials_count * 100
    print(f"success rate : {success_rate}")
    print(f"time : {sum_time}")
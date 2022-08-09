from musicFinder import MusicFinder

'''
객체 생성시 데이터프레임(df_path), 워드투벡(w2v_model_path), 가사임베딩파일(lyrics_vec_path) 필요하며,
같이 생성된 파일로 구성해야 함

w2v_model 은 music_w2v_100_5_sg.model 사용
'''


df_path = "../../data/dataFrame/musicDataFrame_short.json"
w2v_model_path = "../../data/model/music_w2v_100_5_sg.model"
lyrics_vec_path = "../../data/sentVecData/lyrics_vector_data_short_avg.file"


mf = MusicFinder(df_path=df_path, w2v_model_path=w2v_model_path, lyrics_vec_path=lyrics_vec_path)

sentence = ""
while sentence != "0":
    sentence = input("Enter sentence : ")
    music_list = mf.find_music(sentence, topn=15)        # TODO MusicFinder 객체 find_music 함수에 sentence 집어넣으면 딕셔너리 리스트 반환

    for music in music_list:
        music_name = music["musicName"]
        artists = music["artists"]
        sent_idx_list = music["simSentIdx"]
        simSent = [music["lyrics"][sentIdx] for sentIdx in sent_idx_list]

        print(f"music \t\t: {music_name}")
        print(f"artists \t: {artists}")
        print(f"비슷한 가사 \t: {simSent}\n")
import json
import pandas as pd
from vectorSearcher import VectorSearcher
from bin.sentenceEmbedding.sent2vec import Sent2vec as S2V
from konlpy.tag import Okt

class MusicFinder():
    def __init__(
            self, df_path="../../data/dataFrame/musicDataFrame_300000.json", w2v_model_path="../../data/model/music_w2v_100_5_sg.model"
            , lyrics_vec_path = "../../data/sentVecData/lyrics_vector_data_300000_avg.file"
    ):
        self.s2v = S2V(w2v_model_path)
        self.okt = Okt()
        self.lyricsVS = VectorSearcher.load_word2vec_format(lyrics_vec_path)
        self.df = self.load_dataframe(df_path)


    def load_dataframe(self, path) :
        print("[dataFrame]")
        print("\topen dataFrame...")
        with open(path, 'r') as f:
            data = json.loads(f.read())
            del f
        print("\topen complete")

        print("\tmake table...")
        df = pd.DataFrame(data)
        del data

        return df

    def find_music(self, sentence, topn=10):
        tokenized_sentence = self.okt.morphs(sentence, stem=True)
        print(tokenized_sentence)

        sent_vec_topn = topn * 5
        target_vec = self.s2v.sent2vec(tokenized_sentence)
        sim_vec_list = self.lyricsVS.most_similar(target_vec, topn=sent_vec_topn)

        sim_music_list = []
        sim_musicNum_list = []
        for sim_vec in sim_vec_list:
            music_idx, sent_idx = [int(x) for x in sim_vec[0].split(",")]

            sim_music = self.df.loc[str(music_idx)].copy()
            sim_music_num = int(sim_music["musicNum"])

            if sim_music["musicNum"] in sim_musicNum_list:
                for idx in range(len(sim_musicNum_list)):
                    if sim_musicNum_list[idx] == sim_music_num:
                        sim_music_list[idx]["simSentIdx"].append(sent_idx)

            elif len(sim_musicNum_list) < topn:
                sim_musicNum_list.append(sim_music_num)
                sim_music["simSentIdx"] = [sent_idx]
                sim_music_list.append(sim_music)

        return sim_music_list
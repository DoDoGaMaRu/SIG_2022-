from sen2vec import sen2vec
from konlpy.tag import Okt

okt = Okt()

sentence = ""
while sentence != "0":
    sentence = input("Enter : ")
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화

    print(tokenized_sentence)
    print(sen2vec(tokenized_sentence))
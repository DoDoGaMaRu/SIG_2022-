from sen2vec import sen2vec
from konlpy.tag import Okt
from numpy import dot
from numpy.linalg import norm

def cos_sim(vector1, vector2):
    if len(vector1) != len(vector2):
        return 0

    ans = dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    return ans



sentence1 = ""
okt = Okt()
while sentence1 != "0":
    sentence1 = input("Enter sentence1 : ")
    sentence2 = input("Enter sentence2 : ")
    tokenized_sentence1 = okt.morphs(sentence1, stem=True) # 토큰화
    tokenized_sentence2 = okt.morphs(sentence2, stem=True)

    vec1 = sen2vec(tokenized_sentence1)
    vec2 = sen2vec(tokenized_sentence2)

    print("유사도 : " + str(cos_sim(vec1, vec2)))
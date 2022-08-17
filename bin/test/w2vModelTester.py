from gensim.models import KeyedVectors

print("[load model...]")
model = KeyedVectors.load_word2vec_format("../../data/model/music_w2v_100_5_sg.model") #C:\Users\백대환\Desktop\IdeaProjects\SIG_2022하계\data\model\music_w2v
print("\tcomplete")


i = ""
while i != "0":
    i = input("검색어 : ")
    print("[finding...]")

    try:
        model_result = model.most_similar(i)
        print("\tsimilar words")
        for word in model_result:
            print("\t" + str(word))
    except KeyError:
        print("\tnone find")
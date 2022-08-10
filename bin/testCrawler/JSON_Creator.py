import json

path = "../../data/list//"

def saveToFile(name, musics):
    with open(path + name + ".json", 'w', encoding="utf-8") as f:
        json.dump(musics, f, ensure_ascii=False)
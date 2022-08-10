import json

list = []
data1 = ""
data2 = ""

with open("../../data/list/musicList1.json", 'r', encoding="utf-8") as f:
    data1 = json.load(f)

with open("../../data/list/musicList2.json", 'r', encoding="utf-8") as f:
    data2 = json.load(f)


for m1 in data1:
    list.append(m1)

for m2 in data2:
    list.append(m2)

with open("../../data/list/musicList.json", 'w', encoding="utf-8") as f:
    json.dump(list, f, ensure_ascii=False)
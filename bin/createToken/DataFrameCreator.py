import pandas as pd
import json

print("open...")
with open("../../data/list/musicList2.json", 'r', encoding="utf-8") as f:
    data = json.loads(f.read())
print("make table...")
# 35만개로 개수 조절
train_data = pd.DataFrame(data[142571:])

print("개수 : " + str(len(train_data)))

print("한글, 영어 외 문자 제거, 문장분리...")
train_data['lyrics'] = train_data['lyrics'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9 \n]","").str.split("\n")
train_data = train_data.rename(columns={'MusicName':'musicName'})

print("데이터 프레임 저장")
train_data.to_json("../../data/dataFrame//musicDataFrame_300000.json")

print(train_data)
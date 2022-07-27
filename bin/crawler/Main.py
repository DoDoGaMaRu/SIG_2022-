from LiricsCoKrCrawler import *
from JSON_Creator import *
from tqdm import tqdm

list = []

name = "musicList"
start = 0
end = lastMusicNum()

for musicNum in tqdm(range(start, end)):
    music = getMusic(musicNum)
    if music :
        list.append(music)

saveToFile(str(name), list)
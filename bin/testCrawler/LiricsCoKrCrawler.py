import string
import requests
from bs4 import BeautifulSoup

def createSoup(url):
    try:
        html = requests.get(url)
        soup = BeautifulSoup(html.content, 'html.parser')
    except requests.ConnectionError:
        soup = None
    except Exception:
        soup = createSoup(url)

    return soup

def lastMusicNum():
    url = "https://www.lyrics.co.kr/#gsc.tab=0"
    soup = createSoup(url)

    lastMusicUrl = soup.find('div', class_="post-content mt-0").select('a')[0]['href']
    lastMusicNum = int(lastMusicUrl[4:])

    return lastMusicNum

def getMusic(musicNum):
    try:
        url = "https://www.lyrics.co.kr/?p=" + str(musicNum) + "#gsc.tab=0"
        soup = createSoup(url)

        lyricsSoup = soup.find('div', style="font-size: 22px;word-break:break-all;")
        for br in lyricsSoup.find_all("br"):
            br.replace_with("\n")

        title = soup.find('a', class_="post-title mb-2").getText().split("(+)")
        artists = title[0].strip()
        MusicName = title[1].strip()
        lyrics = lyricsSoup.getText().strip()

        answer = {"musicNum":musicNum, 'artists':artists, "musicName":MusicName, "lyrics":cleansing(lyrics)}
    except AttributeError:
        answer = None

    return answer

def cleansing(lyrics):
    return lyrics.translate(str.maketrans('', '', string.punctuation))
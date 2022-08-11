import re
import requests
from bs4 import BeautifulSoup

def _create_soup(url):
    try:
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
    except requests.ConnectionError:
        soup = None

    return soup

class YoutubeCrawler:
    def __init__(self):
        self.youtube_soup = None
        self.video_url = None
        self.video_thumbnail = None

    def search(self, search_sentence):
        url = "https://www.youtube.com/results?search_query=" + search_sentence.replace(" ","+")
        self.youtube_soup = _create_soup(url)
        script_list = self.youtube_soup.find_all('script', nonce=True)

        max_len = 0
        vidio_list_str = ""
        for idx in range(len(script_list)):
            current_str = str(script_list[idx])
            current_len = len(current_str)

            if current_len > max_len:
                max_len = current_len
                vidio_list_str = current_str
        try:
            self.video_url = "https://www.youtube.com" + re.findall(r'"webCommandMetadata":\{"url":"(.+?)"', vidio_list_str, re.S)[1]
        except Exception:
            self.video_url = ""

        try:
            self.video_thumbnail = re.findall(r'"thumbnails":\[\{"url":"(.+?)"', vidio_list_str, re.S)[0]
        except Exception:
            self.video_thumbnail = ""

    def get_video_url(self):
        return self.video_url

    def get_thumbnail_url(self):
        return self.video_thumbnail
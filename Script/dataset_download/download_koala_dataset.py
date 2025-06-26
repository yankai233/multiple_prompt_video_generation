from pytube.cli import on_progress
from pytubefix import YouTube
import moviepy


video = "https://youtu.be/NI9LXzo0UY0?si=_Jy_CzN-QwI4SOkC"
def Download(link):
    youtube_object = YouTube(link)
    youtube_object = youtube_object.streams.get_highest_resolution()
    try:
        youtube_object.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")


Download(video)
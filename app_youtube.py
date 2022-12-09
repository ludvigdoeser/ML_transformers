from pytube import YouTube
from transformers import pipeline
import gradio as gr

pipe = pipeline(model="LudvigDoeser/swedish_asr_model_training")  # change to "your-username/the-name-you-picked"
description = """
<center><img src="https://raw.githubusercontent.com/ludvigdoeser/ML_transformers/main/images/voice_search.png" width=400px></center>
Taligenkänning är ett program som översätter tal till text. Det innebär att du kan använda din röst för att producera text istället för att skriva med tangentbordet! Testa här g$
"""

# news in basic swedish on youtube is a good test

def url2text(link):
    yt = YouTube(link)
    audio = yt.streams.filter(only_audio=True)[0].download(filename="audio.mp4")
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=url2text,
    inputs="text",
    outputs="text",
    title="Svensk Taligenkänning baserad på Whisper Model",
    description=description,
)

iface.launch()

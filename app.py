import gradio as gr
from transformers import pipeline

pipe = pipeline(model="LudvigDoeser/swedish_asr_model_training")  # change to "your-username/the-name-you-picked"
description = """
<center><img src="https://raw.githubusercontent.com/ludvigdoeser/ML_transformers/main/images/voice_search.png" width=400px></center>
Taligenkänning är ett program som översätter tal till text. Det innebär att du kan använda din röst för att producera text istället för att skriva med tangentbordet! Testa här genom att starta inspelningen. När du pratat klart, tryck på avsluta inspelning och sen 'submit'.
"""

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Svensk Taligenkänning baserad på Whisper Model",
    description=description,
)

iface.launch()

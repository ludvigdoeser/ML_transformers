from transformers import pipeline

pipe = pipeline(model="LudvigDoeser/swedish_asr_model_training")  # change to "your-username/the-name-you-picked"

import gradio as gr

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe, 
    inputs=gr.Audio(source="microphone", type="filepath"), 
    outputs="text",
    title="Whisper Small Swedish",
    description="Realtime demo for Swedish speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()

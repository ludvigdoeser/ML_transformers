# ML_transformers
Lab2 in Advanced Scalable Machine Learning Course at KTH

# Intro
Whisper is a pre-trained model for automatic speech recognition (ASR) published in September 2022 by the authors Alec Radford et al. from OpenAI. Unlike many of its predecessors, such as Wav2Vec 2.0, which are pre-trained on un-labelled audio data, Whisper is pre-trained on a vast quantity of labelled audio-transcription data, 680,000 hours to be precise.

In this lab we fine-tune the model with Swedish and communicate the value of our model with an app/service that uses the ML model. Try this link.

# Structure of code

In the file `feature-pipeline.py`, a feature pipeline is defined for loading the Swedish dataset, processesing it, and then storing the data (either at hopsworks or google drive). 

In the file `training-pipeline.py`, the ASR model can be GPU-trained with different settings (e.g. with different optimizers).

The file `app.py` downloads our model from Huggingface, and provides a User Interface to allow users to use their microphone to transcribe Swedish audio to text.

## If we manage:

In the file `feature_daily.py`, we grab a youtube video? 

In the file `batch-inference-pipeline.py`, a batch inference pipeline is built to transcribe the audio.

The file named `app_youtube_video.py` builds a Gradio application to transcribe a video to text.

# Requirements for running the code

Download the repo, create a new conda environment and then run:

```
pip install -r requirements.txt
```

For the interested reader, one can create a pip3 compatible `requirements.txt` file using:

```
pip3 freeze > requirements.txt  # Python3
```

# Evaluation Metric Training

The evaluation metric used is the **Word error rate** (`WER`), which is a common metric of the performance of an automatic speech recognition system. The value indicates the average number of errors per reference word. This means that the lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score.

# Training

## Optimization choices

Try AdamW and adagrad because of no-free-lunch theorem... I will elaborate on this. 

* Last with AdamW:

{'eval_loss': 0.44238531589508057, 'eval_wer': 19.942996961630502, 'eval_runtime': 980.543, 'eval_samples_per_second': 5.17, 'eval_steps_per_second': 0.647, 'epoch': 12.94}
{'train_runtime': 35079.1076, 'train_samples_per_second': 4.561, 'train_steps_per_second': 0.285, 'train_loss': 0.01170906912391074, 'epoch': 12.94} 

* Last with Adagrad:




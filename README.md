# ML_transformers
Lab2 in Advanced Scalable Machine Learning Course at KTH

# Intro
Whisper is a pre-trained model for automatic speech recognition (ASR) published in September 2022 by the authors Alec Radford et al. from OpenAI. Unlike many of its predecessors, such as Wav2Vec 2.0, which are pre-trained on un-labelled audio data, Whisper is pre-trained on a vast quantity of labelled audio-transcription data, 680,000 hours to be precise.

# Requirements

Download the repo, create a new conda environment and then run:

```
pip install -r requirements.txt
```

For the interested reader, one can create a pip3 compatible `requirements.txt` file using:

```
pip3 freeze > requirements.txt  # Python3
```

# Training results

Try AdamW and adagrad because of no-free-lunch theorem... I will elaborate on this. 

Using AdamW these are the model performance results::

Last:

{'eval_loss': 0.44238531589508057, 'eval_wer': 19.942996961630502, 'eval_runtime': 980.543, 'eval_samples_per_second': 5.17, 'eval_steps_per_second': 0.647, 'epoch': 12.94}
{'train_runtime': 35079.1076, 'train_samples_per_second': 4.561, 'train_steps_per_second': 0.285, 'train_loss': 0.01170906912391074, 'epoch': 12.94} 

Best: 
{'eval_loss': 0.3693341314792633, 'eval_wer': 21.228254147508807, 'eval_runtime': 979.4692, 'eval_samples_per_second': 5.175, 'eval_steps_per_second': 0.647, 'epoch': 3.88}

# ML_transformers
Group members: Ludvig Doeser and Xin Huang

Lab2 in Advanced Scalable Machine Learning Course at KTH

# Intro
Whisper is a pre-trained model for automatic speech recognition (ASR) published in September 2022 by the authors Alec Radford et al. from OpenAI. Unlike many of its predecessors, such as Wav2Vec 2.0, which are pre-trained on un-labelled audio data, Whisper is pre-trained on a vast quantity of labelled audio-transcription data, 680,000 hours to be precise.

In this lab we fine-tune the model with Swedish and communicate the value of our model with an app/service that uses the ML model. Try this link.

# Structure of code

In the file `feature-pipeline.py`, a feature pipeline is defined for loading the Swedish dataset, processesing it, and then storing the data (either at hopsworks or google drive). 

In the file `training-pipeline.py`, the ASR model can be GPU-trained with different settings (e.g. with different optimizers).

The file `app.py` downloads our model from Huggingface, and provides a User Interface to allow users to use their microphone to transcribe Swedish audio to text.

The file named `app_youtube.py` builds a Gradio application to transcribe a video (the first 30 seconds at the moment) to text.

# Use the UI

Huggingface Spaces provide us with the possibility of putting our UI there, using this [link](https://huggingface.co/spaces/LudvigDoeser/svenska_taligenkanning)

**Note:** Unfortunately, for some reason it does not work as expected... When running the same `app.py` scripts locally, we get much better results. For this, one has to clone the repo and start a fresh conda environment with the requirement list. Then one can do:

```python
python3 app.py
# or
python3 app_youtube.py #e.g. use https://www.youtube.com/watch?v=RQEMxtM2_X8 (news in basic Swedish)
```

# Requirements for running the code

Download the repo, create a new conda environment and then run:

```
pip install -r requirements.txt
```

For the interested reader, one can create a pip3 compatible `requirements.txt` file using:

```
pip3 freeze > requirements.txt  # Python3
```

# Training

## Evaluation Metric

The evaluation metric used is the **Word error rate** (`WER`), which is a common metric of the performance of an automatic speech recognition system. The value indicates the average number of errors per reference word. This means that the lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score.

## Future improvements on model performance

### Model-centric approach

*

*

A related approach for potential improvement is **choosing the optimal optimizer**. In that spirit, we tried both *AdamW* and *Adagrad*. 
The reason behind trying different optimizers is the no-free-lunch theorem, which states that there is no single best optimization algorithm.

* Best with AdamW:

Best: {'eval_loss': 0.4340263605117798, 'eval_wer': 19.932241671372104, 'eval_runtime': 978.9502, 'eval_samples_per_second': 5.178, 'eval_steps_per_second': 0.648, 'epoch': 10.35}
Last: {'eval_loss': 0.44238531589508057, 'eval_wer': 19.942996961630502, 'eval_runtime': 980.543, 'eval_samples_per_second': 5.17, 'eval_steps_per_second': 0.647, $

{'train_runtime': 35079.1076, 'train_samples_per_second': 4.561, 'train_steps_per_second': 0.285, 'train_loss': 0.01170906912391074, 'epoch': 12.94}

*Note: epoch 12.94 should actually be 13.94 as 1000 steps were trained in another run and this training continued from there*

* Best with Adagrad:

Best: {'eval_loss': 0.4000212848186493, 'eval_wer': 19.942996961630502, 'eval_runtime': 1024.9109, 'eval_samples_per_second': 4.946, 'eval_steps_per_second': 0.619$
Last: {'eval_loss': 0.4368054270744324, 'eval_wer': 19.97257400984109, 'eval_runtime': 1029.183, 'eval_samples_per_second': 4.925, 'eval_steps_per_second': 0.616, 'epoch': 12.94}

{'train_runtime': 38430.0906, 'train_samples_per_second': 4.163, 'train_steps_per_second': 0.26, 'train_loss': 0.0801033290736843, 'epoch': 12.94}

In summary, we see that Adagrad actually results in a lower eval_loss, although AdamW results in a lower eval_wer. Overall, though, the optimizers yield very similar results and no major difference is observed.

### Data-centric approach

New data sources might enable training a better model than the one provided in the blog post. Here are a couple of suggestions:

* 

* Fine-tune the Whisper model using the Norwegian dataset (as this will get us closer to Swedish than the pre-trained model). Then fine-tune again with the Swedish dataset.

### Optimization choices

Try AdamW and adagrad because of no-free-lunch theorem... I will elaborate on this. 

* Best with AdamW:

{'eval_loss': 0.44238531589508057, 'eval_wer': 19.942996961630502, 'eval_runtime': 980.543, 'eval_samples_per_second': 5.17, 'eval_steps_per_second': 0.647, 'epoch': 12.94}
{'train_runtime': 35079.1076, 'train_samples_per_second': 4.561, 'train_steps_per_second': 0.285, 'train_loss': 0.01170906912391074, 'epoch': 12.94} 

* Best with Adagrad:

{'eval_loss': 0.4000212848186493, 'eval_wer': 19.942996961630502, 'eval_runtime': 1024.9109, 'eval_samples_per_second': 4.946, 'eval_steps_per_second': 0.619, 'epoch': 7.76}



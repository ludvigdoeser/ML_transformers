import os
from datasets import list_datasets, load_dataset, DatasetDict
import evaluate

from huggingface_hub import notebook_login
import hopsworks
    
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

import torch
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false' #make sure JAX does not eat up all memory 
torch.cuda.empty_cache()

## ---------------------------------------------------------------------------------------------------------

check_GPU = False
download_data_locally = False # only needs to be done once if you have access to the same machine later on  
data_location = 'GoogleDrive'
google_drive_url = "https://drive.google.com/drive/folders/1yHWh1FxJZswpc4GwukgrrgVcY-bwqF5L?usp=share_link" 
push_to_hub = False # don't have git lfs installed on cluster and don't have access to install it...
output_dir = "./swedish_asr_model_training" #save the model checkpoints here

# Start training from pre-trained model.
# use "openai/whisper-small" for first run
# 2022-12-06: "LudvigDoeser/swedish_asr_model_training" has been trained with the below for 1000 steps; continue there
start_training_from = "LudvigDoeser/swedish_asr_model_training" 

## ---------------------------------------------------------------------------------------------------------

# Login to huggingface and hopsworks
notebook_login()
project = hopsworks.login()

## ---------------------------------------------------------------------------------------------------------

# Check if GPU is available; only works in notebook environment     
def print_GPU_info():
    """
    gpu_info = !nvidia-smi
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
        print('Not connected to a GPU')
    else:
        print(gpu_info)
    """
    pass

if check_GPU:
    print_GPU_info()

# Download the dataset locally
if download_data_locally:
    
    # Hopsworks does not work because of a connection time-out after 15minutes... 
    # but this is the the code that otherwise should work: 
    if data_location=='hopsworks':
        
        # Connect to hopsworks
        dataset_api = project.get_dataset_api() 
        
        # Upload all of the local files to the common_voice directory you created in Hopsworks
        root = "ASR_lab2/common_voice" # path to your hopsworks file directory 
        
        # Have to do it file by file
        hopsworks_paths = [root+"/dataset_dict.json",
                       root+"train/dataset_info.json",
                       root+"train/state.json",
                       root+"train/dataset.arrow/dataset.arrow",
                       root+"test/dataset_info.json",
                       root+"test/state.json",
                       root+"test/dataset.arrow/dataset.arrow",
                      ]
        download_paths =["common_voice/",
                       "common_voice/train/",
                       "common_voice/train/",
                       "common_voice/train/",
                       "common_voice/test/",
                       "common_voice/test/",
                       "common_voice/test/",
                      ]
        
        # Create directory if has not been done: 
        try:
            os.mkdir("common_voice")
            os.mkdir("common_voice/train")
            os.mkdir("common_voice/test")
        except:
            pass 
        
        # Download
        for hp,dp in zip(hopsworks_paths,download_paths):
            print('Hopsworks path = ',hp)
            print('Download path = ',dp)
            downloaded_file_path = dataset_api.download(hp, local_path=dp, overwrite=True)
        
    # Download a folder from google drive
    # Note that this only works ~once every day for large files because google does not want people to store too much data
    elif data_location=="GoogleDrive":
        import gdown
        gdown.download_folder(google_drive_url, quiet=False, use_cookies=False) 

# Print what data is available locally
dir_paths = ['common_voice/','common_voice/train/','common_voice/test/'] # folder path

# Iterate directory
for dir_path in dir_paths:
    print('Checking files in directory: {}'.format(dir_path))
    res = [] # list to store files
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    print('Found files: {} \n'.format(res))

# only works in notebook environment:
#!ls common_voice/
#!ls -lh common_voice/train/
#!ls -lh common_voice/test/

# Connect to hopsworks
dataset_api = project.get_dataset_api()

# Load the downloaded Hugging Face dataset from local disk 
print('Loading training data')
cc = DatasetDict.load_from_disk("common_voice")

# We can leverage the WhisperProcessor we defined earlier to perform both the feature extractor and the tokenizer operations:
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")
# To simplify using the feature extractor and tokenizer, we can wrap both into a single WhisperProcessor class
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")


# Data collecting

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Evaluate 
print('Loading evaluation metric')
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# Load the pre-trained model
print('Loading the pre-trained model')
model = WhisperForConditionalGeneration.from_pretrained(start_training_from)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Prepare training:
print('Defining training arguments')
training_args = Seq2SeqTrainingArguments(
    num_train_epochs=1,
    output_dir=output_dir,  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=10000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=push_to_hub,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=cc["train"], #common_voice['train']
    eval_dataset=cc["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print('Saving pretrained arguments')
processor.save_pretrained(training_args.output_dir)

# Train it 
print('Start training')
trainer.train()
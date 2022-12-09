import hopsworks
from huggingface_hub import notebook_login
from datasets import load_dataset, DatasetDict

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

from datasets import Audio

upload_to = 'googleDrive'

# Login to huggingface and hopsworks
notebook_login()
project = hopsworks.login()

# Load datasets
common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", 
                                     "sv-SE", 
                                     split="train+validation", 
                                     use_auth_token=True)

common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", 
                                    "sv-SE", 
                                    split="test", 
                                    use_auth_token=True)

# Use pre-trained model "Whisper"
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
# The tokenizer maps each of these token ids to their corresponding text string. 
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Swedish", task="transcribe")

# Get rid off some columns
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# Since our input audio is sampled at 48kHz, we need to downsample it to 16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model. We'll set the audio inputs to the correct sampling rate using dataset's cast_column method. This operation does not change the audio in-place, but rather signals to datasets to resample audio samples on the fly the first time that they are loaded:
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# Now we can write a function to prepare our data ready for the model:
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# We can apply the data preparation function to all of our training examples using dataset's .map method. 
common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

# What's in common_voice now? 
help(common_voice)

if upload_to == 'hopsworks':
    print('Uploading data to Hopsworks')
    
    # Upload to hopsworks
    dataset_api = project.get_dataset_api()

    # Upload all of the local files to the common_voice directory you created in Hopsworks
    local_paths = ["./common_voice/dataset_dict.json",
                   "./common_voice/train/dataset_info.json",
                   "./common_voice/train/state.json",
                   "./common_voice/train/dataset.arrow"
                   "./common_voice/test/dataset_info.json",
                   "./common_voice/test/state.json",
                   "./common_voice/test/dataset.arrow",
                  ]
    upload_paths =["common_voice/",
                   "common_voice/train/",
                   "common_voice/train/dataset.arrow" 
                   "common_voice/train/",
                   "common_voice/test/",
                   "common_voice/test/",
                   "common_voice/test/dataset.arrow",
                  ]

    for lp,up in zip(local_paths,upload_paths):
        uploaded_file_path = dataset_api.upload(local_path = lp, upload_path = up, overwrite=True)

elif upload_to == 'googleDrive': #only works in google notebook environment
    print('Uploading data to Google Drive')
    
    from google.colab import drive
    drive.mount('/content/gdrive')
    try:
        os.mkdir("/content/gdrive/MyDrive/MLdata/common_voice")
    except:
        pass
    common_voice.save_to_disk(F"/content/gdrive/MyDrive/MLdata/common_voice/")

else:
    print('Not uploading data anywhere')
    pass
    
print('Done')

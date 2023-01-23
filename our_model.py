
#pip install -r  requirements.txt

from IPython.display import Javascript
from base64 import b64decode
from io import BytesIO
import pandas as pd
from datasets import Dataset
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import pyaudio
import wave



### recording function##
def record(sec=6):
    # Record in chunks of 1024 samples
    chunk = 1024 
    # 16 bits per sample
    sample_format = pyaudio.paInt16 
    chanels = 1
    # Record at 44400 samples per second
    smpl_rt = 44400 
    seconds = sec
    filename = "output.wav"
    # Create an interface to PortAudio
    pa = pyaudio.PyAudio() 
    stream = pa.open(format=sample_format, channels=chanels,
                    rate=smpl_rt, input=True,
                    frames_per_buffer=chunk)
    print('Recording...')
    frames = [] 
    # Store data in chunks for 8 seconds
    for i in range(0, int(smpl_rt / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate - PortAudio interface
    pa.terminate()
    print('Done !!! ')
    # Save the recorded data in a .wav format
    sf = wave.open(filename, 'wb')
    sf.setnchannels(chanels)
    sf.setsampwidth(pa.get_sample_size(sample_format))
    sf.setframerate(smpl_rt)
    sf.writeframes(b''.join(frames))
    sf.close()
####



## Adjust the path of the model 
'''
get the model from this drive link https://drive.google.com/drive/folders/1-LKd7eWYIHC0XNQTZoer2o_3kpamZgyB?usp=share_link
and add pass the path to the model below

get the processor from the this drive link https://drive.google.com/drive/folders/11HhaXk0_bBHz_XI6KdTZ8U1vJhQp2rJz?usp=share_link
and add pass the path to the processor below
'''

model = Wav2Vec2ForCTC.from_pretrained("G:\Year5_fall\math pro\project/checkpoint-17700") 
processor = Wav2Vec2Processor.from_pretrained("G:\Year5_fall\math pro\project/final_processor")



def speech_file_to_array_fn(batch):
    
    speech_array, sampling_rate = torchaudio.load(batch["audio"])
    resampler = torchaudio.transforms.Resample(sampling_rate, 16_000) 
    batch["audio"] = resampler(speech_array).squeeze().numpy()
    return batch




def specch_to_text(audiofile='',rec=0):
  if audiofile!='':
    path=audiofile
  elif rec:
       print("recording start for 5 second,speak please")
       record(sec=5)
       path="output.wav"
  else :
        print("no audio file detected ")
        return
  dftest = pd.DataFrame(columns=['audio'])
  dftest['audio']=[path]  
  test_dataset1 = Dataset.from_pandas(dftest)

  test_dataset = test_dataset1.map(speech_file_to_array_fn)
  inputs = processor(test_dataset["audio"], sampling_rate=16_000, return_tensors="pt", padding=True)

  with torch.no_grad():
      logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

  predicted_ids = torch.argmax(logits, dim=-1)
  pred=processor.batch_decode(predicted_ids)
  pred= pred[0].replace('ى','ي')
  return pred
  


## Run this function to return prediction
print(specch_to_text(audiofile='',rec=1))


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




### recording function##
RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""
def record(sec=3):
  display(Javascript(RECORD))
  s = output.eval_js('record(%d)' % (sec*1000))
  b = b64decode(s.split(',')[1])
  audio = AudioSegment.from_file(BytesIO(b))
  return audio

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




def specch_to_text(audiofile='',record=0):
  if audiofile!='':
    path=audiofile
  elif record:
       print("recording start for 5 second,speak please")
       audio=record(sec=5)
       audio.export(out_f = "output.wav",
                        format = "wav")
       path="output.wav"
  else :
      "no audio file detected "
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
print(specch_to_text(audiofile='G:\Year5_fall\math pro/3255.wav',record=0))
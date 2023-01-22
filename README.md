# tuning_XLSR-Wav2Vec2-_on_Egyptian_dialect-and-emotion-detection

our work
Wav2Vec is trained on a large dataset of speech samples and is able to extract high-level features from the speech signal that are useful for both emotion recognition and speech recognition tasks. The proposed approach is evaluated on a benchmark dataset of spoken emotions and shows promising results, outperforming existing methods in both emotion recognition and speech recognition tasks[2]. The approach is also shown to be robust to variations in speaking styles and environmental conditions. We  trained our speech recognition model on a dataset consisting of 5.5 hours which is 3041 audio.
We achieved a word error rate(WER) of 33%. For emotion recognition, we used Egyptian Arabic seminatural emotional speech (EYASE) database created from an award winning Egyptian drama series. 
We achieved accuracy for emotion reaching 74%. 


Download the model and processor from the following links:
get the model from this drive link https://drive.google.com/drive/folders/1-LKd7eWYIHC0XNQTZoer2o_3kpamZgyB?usp=share_link
and add pass the path to the model below

get the processor from the this drive link https://drive.google.com/drive/folders/11HhaXk0_bBHz_XI6KdTZ8U1vJhQp2rJz?usp=share_link
and add pass the path to the processor below


To run file our_model.py on the local machine 
1-pip install -r requrirement.txt
2-adjust the paths for the model,processor and audio file inside the file
3-recording also is provided in the file,see function speech_to_text


References:
dataset for emotion detection upon request https://www.researchgate.net/publication/341001383_EYASE_Database
Egytian corpus  https://magichub.com/datasets/egyptian-arabic-conversational-speech-corpus/

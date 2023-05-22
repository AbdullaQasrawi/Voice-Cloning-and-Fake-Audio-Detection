import os
from IPython.display import Audio, clear_output
from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import wavio
import spacy
import random
import pandas as pd
import noisereduce as nr
from scipy.io import wavfile
import speech_recognition as sr
from IPython.display import Audio, clear_output
from moviepy.editor import concatenate_audioclips, AudioFileClip
import warnings

class VoiceCloning:
    def __init__(self):
        # Cloning the repository
        !git clone https://github.com/misbah4064/Real-Time-Voice-Cloning.git
        
        # Installing the dependencies
        !pip install -q -r requirements.txt
        !apt-get install -qq libportaudio2
        
        # Initializing all the encoder libraries
        encoder_weights = Path(r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder (2)\encoder\saved_models\pretrained.pt")
        vocoder_weights = Path(r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder (2)\vocoder\saved_models\pretrained\pretrained.pt")
        syn_dir = Path(r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder (2)\synthesizer\saved_models\logs-pretrained\taco_pretrained")
        encoder.load_model(encoder_weights)
        self.synthesizer = Synthesizer(syn_dir)
        vocoder.load_model(vocoder_weights)
        
        # Other initializations
        self.r = sr.Recognizer()
        self.nlp = spacy.load('en_core_web_sm')
        self.train = pd.read_csv(r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\TIMIT Dataset\train_data.csv")
        self.test = pd.read_csv(r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\TIMIT Dataset\test_data.csv")

        warnings.filterwarnings("ignore")
    
    def synthesize(self, audio, text, audio_name):
        """
        Used to clone the audio
        """
        embedding = encoder.embed_utterance(encoder.preprocess_wav(audio, 22050))
        print("Synthesizing new audio...")
        specs = self.synthesizer.synthesize_spectrograms([text], [embedding])
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = np.pad(generated_wav, (1, self.synthesizer.sample_rate), mode="constant")
        clear_output()
        wavio.write(audio_name, generated_wav, sampwidth=2, rate=self.synthesizer.sample_rate)

    def speech_to_text(self, audio):
        """
        return the text from cloned audio
        """
        with sr.AudioFile(audio) as source:
            # listen for the data (load audio to memory)
            audio_data = self.r.record(source)
            # recognize (convert from speech to text)
            text = self.r.recognize_google(audio_data)
            return text
        

    def text_comparison(self, audio, text, nlp=None):
        nlp = nlp or self.nlp
        new_text = self.speech_to_text(audio)
        original_text = nlp(text)
        return original_text.similarity(nlp(new_text))
       
    def convert_to_long_audio(self):
        # Correct the path
        self.train["path_from_data_dir_windows"] = self.train["path_from_data_dir_windows"].apply(lambda path: "C:/Users/abedq/Desktop/Apziva/Apziva/P6/New folder/TIMIT Dataset/data/" + str(path))
        self.test["path_from_data_dir_windows"] = self.test["path_from_data_dir_windows"].apply(lambda path: "C:/Users/abedq/Desktop/Apziva/Apziva/P6/New folder/TIMIT Dataset/data/" + str(path))

        self.df = pd.concat([self.train, self.test], axis=0)

        # Create speaker dictionary
        speaker_id_dic = {}
        for i in self.df.speaker_id.unique():
            speaker_id_dic[i] = []

        for i in range(self.df.shape[0]):
            for key, value in speaker_id_dic.items():
                if self.df.speaker_id.tolist()[i] == key:
                    if self.df.path_from_data_dir.str.endswith('.wav').tolist()[i]:
                        value.append(self.df.path_from_data_dir_windows[i])



        # Convert the short audio files to long ones for all of the audio files
        for key, value in speaker_id_dic.items():
            files = speaker_id_dic[key]
            clips = [AudioFileClip(c) for c in files]
            final_clip = concatenate_audioclips(clips)
            final_clip.write_audiofile(r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\TIMIT Dataset\long_version_audio\{}.wav".format(key))

    def clone_audio(self):
        self.text_files = [i for i in self.df.path_from_data_dir_windows if i.endswith('TXT')]

        text_list = []
        for file in self.text_files:
            with open(file) as f:
                lines = f.readlines()
                text_list.append(lines)

        final_text = []
        for i in range(len(text_list)):
            text = text_list[i][0].split(' ')[2:]
            text = ' '.join(text)
            text = text.replace('\n', '').replace('.', '').replace('?', '').replace('!', '')  # Important
            final_text.append(text)

        audio_name = [i for i in os.listdir(r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\TIMIT Dataset\long_version_audio")]

        cloned_df = pd.DataFrame(audio_name, columns=['file'])
        cloned_df['text'] = ""

        for i in range(cloned_df.shape[0]):
            cloned_df.at[i, 'text'] = random.choice(list(set(final_text)))

        cloned_df.to_csv(r'C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\TIMIT Dataset\cloned_df.csv')

        counter = 1
        for i in range(cloned_df.shape[0]):
            text = cloned_df.at[i, 'text']
            file_path = r"C:\Users\abedq\Desktop\Apziva\Apziva\P6\New folder\TIMIT Dataset\long_version_audio\\" + cloned_df.at[i, 'file']
            self.synthesize(file_path, text, file_path)
            print(r"{} files have been cloned out of 336".format(counter))
            counter +=1


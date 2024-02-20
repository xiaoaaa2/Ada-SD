from macls.data_utils.reader_copy import CustomDataset
from macls.data_utils.audio import AudioSegment
import soundfile as sf
import random
import os


audio_path = '/home/data_aishell/wav/train'
write_clean_wav_path = '/home/dataset/audio/clean/'
filenames=os.listdir(audio_path)[:200]
print(filenames)
n=0
for filename in filenames:   # Generate clean speech segment data
    path = os.path.join(audio_path, filename)
    filenames=os.listdir(path)[:35]
    for filename2 in filenames:  
        n+=1
        path2 = os.path.join(path, filename2)
        m1_duration = random.uniform(2.5, 4.5)
        audio_segment = AudioSegment.from_file(path2)
        audio_segment.crop(duration=m1_duration, mode='train')
        write_wav_path = os.path.join(write_clean_wav_path,f'{n}.wav')
        sf.write(write_wav_path,audio_segment.samples,16000,format='WAV',subtype='FLOAT')


audio1_path = '/home/audio_clean/clean1'   # The path to store 6000 clean speech segments
audio2_path = '/home/audio_clean/clean2'   # The path to store different 6000 clean speech segments
write_mixed_wav_path = '/home/zidonghua/zcy/AudioClassification-Pytorch/dataset/audio/mix/'
filenames1=os.listdir(audio_path)[0: 200]
filenames2=os.listdir(audio_path)[200: 400]
n=0
for filename1, filename2 in zip(filenames1, filenames2):   # Generate mixed speech segment data
    n+=1 
    path1 = os.path.join(audio_path, filename1)
    path2 = os.path.join(audio_path, filename2)
    m1_duration = random.uniform(0.8, 1.75)
    duration = random.uniform(2.5, 4.5)
    m2_duration = duration - m1_duration
    audio_segment1 = AudioSegment.from_file(path1)
    audio_segment1.crop(duration=m1_duration, mode='train')
    audio_segment2 = AudioSegment.from_file(path2)
    audio_segment2.crop(duration=m2_duration, mode='train')
  
    # Set the duration of fade-in and fade-out
    fade_duration = 1000  # 1 second
    # Add fade-in effect
    audio_fadein = audio_segment2.fade_in(fade_duration)
    # Add fade-out effect
    audio_fadeout = audio_segment1.fade_out(fade_duration)
  
    # Concatenate two audio segments
    audio = AudioSegment.concatenate(audio_fadein, audio_fadeout)
    write_wav_path = os.path.join(write_mixed_wav_path,f'{n}.wav')
    sf.write(write_wav_path,audio.samples,16000,format='WAV',subtype='FLOAT')


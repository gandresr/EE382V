import librosa
import os

fnames = ['coffee.wav', 'kitchen.wav', 'party.mp3', 'soccer.wav']
fpaths = [os.path.join(os.getcwd(), os.pardir, s) for s in fnames]

for fpath in fpaths:
    y, sr = librosa.load(fpath)
    tempo, beat_frames = librosa.beat.beat_track(y, sr)
    print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    # 4. Convert the frame indices of beat events into timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
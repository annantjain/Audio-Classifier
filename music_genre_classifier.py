import gradio as gr
import numpy as np
import tensorflow as tf
import librosa
import math
from tensorflow.keras.models import load_model

model_path = "music_genre_model2.h5"
model = load_model(model_path)
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
num_segments=10
samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
genre_labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock", "bollypop", "carnatic", "ghazal", "semiclassical", "sufi"]
num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / 512)
def predict_genre(audio_file):
    # Read the audio file content
    audio_data, _ = librosa.load(audio_file.name, sr=SAMPLE_RATE)

    # Ensure audio is the correct duration
    if len(audio_data) >  SAMPLE_RATE * TRACK_DURATION:
        audio_data = audio_data[:SAMPLE_RATE * TRACK_DURATION]
    elif len(audio_data) < SAMPLE_RATE * TRACK_DURATION:
        padding = np.zeros(SAMPLE_RATE * TRACK_DURATION - len(audio_data))
        audio_data = np.concatenate((audio_data, padding))

    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(y=audio_data[start:finish], sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=512)
        print(mfcc.shape)
        print(num_mfcc_vectors_per_segment)
        mfcc = mfcc.T

        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
            predicted_index = np.argmax(model.predict(mfcc), axis=1)
            print(predicted_index)
            predicted_genre = genre_labels[predicted_index[0]]


    return predicted_genre

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.inputs.File(label="Upload an audio file"),
    outputs="text"
)

# Launch the Gradio interface
interface.launch()

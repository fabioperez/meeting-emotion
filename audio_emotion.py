import io
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import requests
import streamlit as st
from pydub import AudioSegment


def huggingface_api(payload, model_id, api_token):
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def query(audio_data):
    API_TOKEN = os.getenv("API_TOKEN")
    model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.request("POST", API_URL, headers=headers, data=audio_data)
    print(response)
    return json.loads(response.content.decode("utf-8"))


def divide_audio_in_chunks(audio_file, chunk_size):
    """
    Divides audio file into chunks of size chunk_size.
    :param audio_file: audio file to be divided
    :param chunk_size: size of each chunk
    :return: list of chunks
    """
    with open(audio_file, encoding="ISO-8859-1") as f:
        audio_bytes = f.read()
    chunks = []
    for i in range(0, len(audio_bytes), chunk_size):
        chunks.append(audio_bytes[i : i + chunk_size])
    return chunks


@st.cache
def get_emotions(audio, snippet_length):
    model_results = []
    for snippet_idx in range(0, len(audio) // snippet_length - 1):
        snippet = (
            audio[snippet_length * (snippet_idx) : snippet_length * (snippet_idx + 1)]
            .export()
            .read()
        )
        result = query(snippet)
        model_results.append(result)
    return model_results


def get_audio_type(audio_bytes):
    pass


if __name__ == "__main__":
    st.title("Meeting Emotion Recognition")
    st.write("This app will help you to recognize the emotions of a meeting.")
    st.write(
        "Upload a file containing the meeting audio (mp3 or wav), and we will extract the emotions from the meeting."
    )
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
    if audio_file:
        audio_bytes = audio_file.read()
        s = io.BytesIO(audio_bytes)
        if audio_file.name.endswith(".wav"):
            audio = AudioSegment.from_file(s, "wav")
        elif audio_file.name.endswith(".mp3"):
            audio = AudioSegment.from_file(s, "mp4")
        st.audio(audio.export().read())

        snippet_length = st.number_input(
            "Audio snippet length (in ms)", min_value=1, max_value=10000, value=2000
        )
        emotions = get_emotions(audio, snippet_length)
        emotions = [
            emotion[0].get("label", "fail") if type(emotion) == list else "error"
            for emotion in emotions
        ]
        emotions_count_df = pd.DataFrame(
            Counter(emotions).items(), columns=["Emotion", "Count"]
        )
        emotions_df = pd.DataFrame(emotions, columns=["Emotion"]).reset_index()
        emotions_df["a"] = "a"
        emotions_df["index"] = emotions_df["index"].astype(int)
        emotions_df["start"] = (emotions_df["index"] * snippet_length) / 1000
        emotions_df["end"] = ((emotions_df["index"] + 1) * snippet_length - 1) / 1000
        emotions_df["delta"] = emotions_df["end"] - emotions_df["start"]
        emotions_df["Task"] = "Meeting"
        emotions_df["Start"] = emotions_df["start"]
        emotions_df["Finish"] = emotions_df["end"]
        # # st.write(emotions_df)
        # st.text(emotions)

        st.header("Emotion distribution")
        st.plotly_chart(px.bar(emotions_count_df, x="Emotion", y="Count"))

        st.header("Emotion timeline")
        # fig = px.timeline(
        #     emotions_df,
        #     x_start='start',
        #     x_end='end',
        #     y='Emotion',
        #     # color='Emotion'
        # )
        # fig.update_yaxes(autorange="reversed")
        # fig.layout.xaxis.type = 'linear'
        # fig.data[0].x = emotions_df.delta.tolist()
        # fig.update_xaxes(title_text='Seconds')

        # st.plotly_chart(fig)

        # colors = {
        #     'surprised': 'yellow',
        #     'angry': 'red',
        #     'sad': 'blue',
        #     'disgust': 'orange',
        #     'fearful': 'purple',
        #     'neutral': 'gray',
        #     'happy': 'green'
        # }
        fig = ff.create_gantt(
            emotions_df,
            index_col="Emotion",
            show_colorbar=True,
            group_tasks=True,
            title="",
            height=600,
            width=800,
            bar_width=1.0,
        )
        fig.layout.xaxis.type = "linear"
        fig.data[0].x = emotions_df.delta.tolist()
        fig.update_xaxes(title_text="Seconds")

        st.plotly_chart(fig)

import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
import librosa
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import plotly.graph_objs as go
import plotly.express as px
import os
import shutil
import io
import warnings
import datetime as dt
import soundfile as sf
import pandas as pd
from process_audio import detect_voice_activity, deepfilter
from main import translation_model, transcription_model
from auth import config, nameList

warnings.filterwarnings('ignore')

UPLOAD_DIR = os.path.join(Path(__file__).parent, "uploaded_audios")
DATABASE_PATH = os.path.join(Path(__file__).parent, "database.pkl")
DATABASE_DIR = os.path.join(Path(__file__).parent, "database_audios")

if not os.path.exists(UPLOAD_DIR):
    os.mkdir(UPLOAD_DIR)

if not os.path.exists(DATABASE_DIR):
    os.mkdir(DATABASE_DIR)

# Function to create waveform plot
def create_waveform_plot(y, sr, highlight_start=None, highlight_end=None, cursor_time=None):
    time_axis = np.arange(len(y)) / sr
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=y, mode='lines', name='Waveform', line=dict(color='red', width=1)))
    
    if highlight_start is not None and highlight_end is not None and highlight_end-highlight_start!=len(y)/sr:
        fig.add_vrect(x0=highlight_start, x1=highlight_end, fillcolor="white", opacity=0.3, line_width=0)
    
    if cursor_time is not None:
        fig.add_vline(x=cursor_time, line=dict(color='white', width=2))

    fig.update_layout(
        height=400, 
        width=800, 
        yaxis=dict(
            title='Amplitude',
            range=[min(y), max(y)],
            fixedrange=True
        ),
        xaxis=dict(
            title='Time (s)'
        ),
        margin=dict(l=0, r=0, t=0, b=30)
    )
    
    return fig

def save_new_audio(uploaded_file):
    """
    Save the uploaded document to the defined directory.
    :param uploaded_file: The file uploaded by the user.
    """
    
    if uploaded_file is not None:
        # Convert the file to WAV format if it's in MP3
        if uploaded_file.type == "audio/mpeg":
            audio = AudioSegment.from_mp3(uploaded_file)
        elif uploaded_file.type == "audio/wav":
            audio = AudioSegment.from_wav(uploaded_file)
        elif uploaded_file.type == "audio/ogg":
            audio = AudioSegment.from_ogg(uploaded_file)
        else:
            st.error("Unsupported file type")
            st.stop()

        # Export audio to WAV format in a buffer
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        # Save the buffer as a WAV file to UPLOAD_DIR
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name.split('.')[0] + '.wav')
        with open(save_path, 'wb') as f:
            f.write(buffer.read())
        buffer.seek(0)  # Reset buffer position for further use
       
    return None

# Set the page config
st.set_page_config(page_title="Translation Assist", layout="wide")
# Title for the Streamlit app
st.title("Translation Assistance Suite")
st.markdown('<style>div.block-container{padding-top:=0rem;}</style>', unsafe_allow_html=True)

tab = option_menu(
    menu_title="",
    # menu_icon='chat-text-fill',
    options=["Translation", "Analysis", "Settings"],
    icons=["translate", "database-fill", "gear"],
    default_index=0,
    orientation='horizontal'
)

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "translation" not in st.session_state:
    st.session_state.translate = ""
if "clipsave" not in st.session_state:
    st.session_state.clipsave = False
if tab=="Translation":
    with st.sidebar:
    # File uploader to upload audio files
        uploaded_files = st.file_uploader("Upload audios", type=["wav", "mp3", "m4a"], accept_multiple_files=True)

        for uploaded_file in uploaded_files:
            save_new_audio(uploaded_file=uploaded_file)
        
        audio = st.selectbox(
            label="Select Audio for processing",
            options=[audio for audio in os.listdir(UPLOAD_DIR) if audio.split(".")[-1] in ["wav", "mp3", 'm4a']],
            disabled=True if len(os.listdir(UPLOAD_DIR))==0 else False
        )
        with st.form(key="delete-key"):
            del_audios = st.multiselect(
                label="Select Audios for deletion",
                options=[audio for audio in os.listdir(UPLOAD_DIR)]
            )

            if st.form_submit_button(label='Delete Files', help="Delete Unused Audios", use_container_width=True):
                for del_audio in del_audios:
                    os.remove(os.path.join(UPLOAD_DIR, del_audio))
                    st.toast(body=f"{del_audio} audio file deleted.")
                st.rerun()

    if audio is not None:
        # Load the audio file
        try:
            y, sr = librosa.load(os.path.join(UPLOAD_DIR, audio), sr=None)
        
            duration = librosa.get_duration(y=y, sr=sr)
            min_secs = str(dt.timedelta(seconds=duration))
            col1, col2 = st.columns([0.7, 0.3], vertical_alignment='top', gap="medium")
            with col1:
                col11, col12 = st.columns(2)
                with col11:
                    start_time = st.number_input("Start Time (seconds)", min_value=0.0, max_value=duration, value=0.0) 
                with col12:
                    end_time = st.number_input("End Time (seconds)", min_value=0.0, max_value=duration, value=duration)

                colblank, colcursor = st.columns([0.05, 0.95])
                with colcursor:
                    cursor = st.slider(label="Cursor", min_value=start_time, max_value=end_time, step=0.01)
                waveform = create_waveform_plot(y=y, sr=sr, highlight_start=start_time, highlight_end=end_time, cursor_time=cursor)
                waveform.update_layout(height=200)
                st.plotly_chart(waveform, use_container_width=True)

                # Crop the selected chunk
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                cropped_audio = y[start_sample:end_sample]

                colaudio, colsave = st.columns([0.95, 0.05], vertical_alignment='center')
                with colaudio:
                    # Play the audio file
                    st.audio(cropped_audio, format="audio/wav", sample_rate=sr)
                with colsave:
                    if st.button(label=":white_check_mark:"):
                        st.session_state.clipsave = True
                        

                coldetect, colnoise, coltranscribe = st.columns(3)
                
                with coldetect:
                    vad_button = st.button(label=":speaking_head_in_silhouette: Detect Speech", use_container_width=True, help="Detect Voice Activity")

                    if vad_button:
                        vad_status = detect_voice_activity(input_audio=str(os.path.join(UPLOAD_DIR, audio)), output_path=os.path.join(UPLOAD_DIR, "vad"+audio))
                        st.toast(body=vad_status['message'])
                        st.rerun()

                with colnoise:
                    clear_button = st.button(label=":speech_balloon: Clean Audio", use_container_width=True, help="Reomve background Noise")

                    if clear_button:
                        clear_status = deepfilter(input_audio=str(os.path.join(UPLOAD_DIR, audio)), output_audio=os.path.join(UPLOAD_DIR, "clean"+audio.replace(audio.split(".")[-1], "wav")))
                        st.toast(body=clear_status['message'])
                        st.rerun()
                
                with coltranscribe:
                    transcribe = st.button(label=":u5272: Transcribe Audio", use_container_width=True, help="Transcribe in mandarin")

                    if transcribe:
                        st.toast(body="Transcription Service Started...")
                        pipe = transcription_model(model='v3')
                        transcript = pipe(os.path.join(UPLOAD_DIR, audio), generate_kwargs={"language":"mandarin"}, return_timestamps=True)
                        st.toast(body="Transcription Completed")
                        st.session_state.transcript = transcript['text']
                        
                    else:
                        pass
                
            with col2:
                st.subheader("Analysis Tab")
                with st.form(key='menu form', border=True):
                    st.write("<h4>Mandarin</h4>", unsafe_allow_html=True)
                    st.markdown("<style>div.text-container{margin-top:1px;}</style>", unsafe_allow_html=True)
                    st.caption(st.session_state.transcript)
                    coltranslate, coltranscript = st.columns(2)
                    with coltranslate:
                        translate = st.form_submit_button(label=":gear: Translate", help="Translate to english", use_container_width=True)
                    with coltranscript:
                        reset = st.form_submit_button(label=":arrows_counterclockwise: Reset", use_container_width=True)


                    if translate:
                        st.toast(body="Translation Service Started...")
                        translation_text = translation_model(st.session_state.transcript if st.session_state.transcript else "", model='llama3')
                        st.session_state.translate = translation_text
                        st.write("<h4>English</h4> Correct as per requirement", unsafe_allow_html=True)
                        translated_data = st.text_area(label="", value=st.session_state.translate, height=500)
                        st.write(type(translated_data))
                        st.toast(body="Translation Completed...")
                    
                    if reset:
                        st.session_state.transcript = ""
                        st.session_state.translate = ""
                        st.rerun()

                    save = st.form_submit_button(label=":inbox_tray: Save and Export", help="Save transcript to database", use_container_width=True)
                    if save:
                        if os.path.exists(DATABASE_PATH):
                            df = pd.read_pickle(DATABASE_PATH)
                            df.loc[len(df), :] = [str(os.path.join(UPLOAD_DIR, audio)), start_time, end_time, st.session_state.transcript, st.session_state.translate]
                            df.to_pickle(DATABASE_PATH)
                            shutil.copy(src=str(os.path.join(UPLOAD_DIR, audio)), dst=str(os.path.join(DATABASE_DIR, audio)))
                            st.toast(body="Saved to database")
                        else:
                            df = pd.DataFrame(columns=['filepath', 'start_time', 'end_time', 'mandarin', 'english'])
                            df.loc[len(df), :] = [str(os.path.join(UPLOAD_DIR, audio)), start_time, end_time, st.session_state.transcript, st.session_state.translate]
                            df.to_pickle(DATABASE_PATH)
                            shutil.copy(src=str(os.path.join(UPLOAD_DIR, audio)), dst=str(os.path.join(DATABASE_DIR, audio)))
                            st.toast(body="Saved to database")
                    

                if st.session_state.clipsave:
                    with st.form(key="save-clip", border=True):
                        coltext, colbtn = st.columns([0.8, 0.2], vertical_alignment='center')
                        with coltext:
                            clip_name = st.text_input(label="Clip Name", help="Enter the namme of the clip", placeholder="Enter the filename without extension")
                        with colbtn:
                            if st.form_submit_button(label="Save"):
                                st.session_state.clipsave = False  
                        
                                if clip_name:
                                    sf.write(file=os.path.join(UPLOAD_DIR, clip_name + '.wav'), data=cropped_audio, samplerate=sr)
                                    st.rerun()
                                else:
                                    sf.write(file=os.path.join(UPLOAD_DIR, f"cropped-{start_time}-{end_time}-" + audio), data=cropped_audio, samplerate=sr)
                                    st.rerun() 


        except FileNotFoundError as e:
            st.error(f"Error: Audio file does ot exist. Please load a fresh audio file")
elif tab=="Analysis":
    st.write("Analysis here")



elif tab=="Settings":
    
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['credentials']['cookie']['name'],
        config['credentials']['cookie']['key'],
        config['credentials']['cookie']['expiry_days']
    )

    authenticator.login()
    
    if st.session_state['authentication_status'] == False:
        st.error("Username/Password is incorrect")

    if st.session_state['authentication_status'] == None:
        st.warning("Please enter username and password")
    
    if st.session_state['authentication_status']:
        colfirst, collast = st.columns([0.9, 0.1])
        with colfirst:
            st.write("Settings")
        with collast:
            if st.session_state['username'] in nameList:
                authenticator.logout("Logout")
                st.markdown("<style>div.logout-container{margin-right:1rem;}</style>", unsafe_allow_html=True)
                    
        with st.sidebar:
            st.write("Sidebar here")
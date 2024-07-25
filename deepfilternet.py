from df.enhance import enhance, init_df, load_audio, save_audio
import librosa
import soundfile as sf

def deepfilter(input_audio:str, output_audio:str, ):

    # Load the MP3 file
    file_path = 'D:/translation-whisper/audios/voice1.mp3'
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Resample to 48 kHz
    target_sample_rate = 48000
    audio_data_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)

    # Save the resampled audio to a WAV file
    output_path = 'D:/translation-whisper/audios/voice1.wav'
    sf.write(output_path, audio_data_resampled, target_sample_rate)

    # Load default model
    model, df_state, _ = init_df()
    # Download and open some audio file. You use your audio files here
    audio_path = output_path
    audio, _ = load_audio(audio_path, sr=df_state.sr())
    # Denoise the audio
    enhanced = enhance(model, df_state, audio)
    # Save for listening
    save_audio("D:/translation-whisper/audios/dfnet-voice1.wav", enhanced, df_state.sr())
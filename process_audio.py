import librosa
import webrtcvad
import numpy as np
import soundfile as sf
import subprocess
from df.enhance import enhance, init_df, load_audio, save_audio


def reencode_audio(input_audio, output_audio):
    print(f"Repairing the audiofile {input_audio}")
    print()
    command = [
        'ffmpeg', '-i', input_audio, '-c:a', 'libmp3lame', '-q:a', '2', output_audio
    ]
    subprocess.run(command, check=True)

def resample_audio(input_audio:str, output_audio:str, save=True):
    # Load the MP3 file
    audio_data, sample_rate = librosa.load(input_audio, sr=None)
    
    print(f"Sample Rate of {sample_rate/1000} KHz detected. Resampling it to 48 KHz...")
    # Resample to 48 kHz
    target_sample_rate = 48000
    audio_data_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
    
    if save:
        # Save the resampled audio to a WAV file
        sf.write(output_audio, audio_data_resampled, target_sample_rate)
        return {"audiofile":output_audio,"audiodata":audio_data_resampled, "message":f"Resampled audio saved at {output_audio}"}
    else:
        return {"audiodata":audio_data_resampled, "message":f"Resampled audio saved at {output_audio}"}


# Function to convert audio frame to the right format for webrtcvad
def frame_generator(audio, sample_rate, frame_duration_ms):
    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

# Function to collect segments with speech activity
def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = []
    triggered = False
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)

        if not triggered:
            ring_buffer.append(frame)
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            if sum([vad.is_speech(f.tobytes(), sample_rate) for f in ring_buffer]) > 0.9 * num_padding_frames:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            if sum([vad.is_speech(f.tobytes(), sample_rate) for f in ring_buffer]) < 0.1 * num_padding_frames:
                triggered = False
                yield b''.join(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        yield b''.join(voiced_frames)

def detect_voice_activity(input_audio:str, output_path:str, sample_rate=48000):
    
    resample = resample_audio(input_audio=input_audio, output_audio="", save=False)
    audio_data = resample['audiodata']
    
    # Convert audio data to 16-bit PCM format
    audio_data_pcm = (audio_data * 32767).astype(np.int16)

    # Initialize webrtcvad
    vad = webrtcvad.Vad(3)  # 0 is least aggressive, 3 is most aggressive

    # Create frames
    frame_duration_ms = 30  # Duration of each frame in ms
    frames = frame_generator(audio_data_pcm, sample_rate, frame_duration_ms)

    # Filter out non-speech frames
    segments = vad_collector(sample_rate, frame_duration_ms, 300, vad, frames)
    filtered_audio_data = np.concatenate([np.frombuffer(segment, dtype=np.int16) for segment in segments])

    # Convert back to floating point format
    filtered_audio_data = filtered_audio_data.astype(np.float32) / 32767

    # Save the result to a new file
    sf.write(output_path, filtered_audio_data, sample_rate)
    return {"audiofile":output_path, "message":f"Voice activity detected and trimmed audio saved at {output_path}"}

def deepfilter(input_audio:str, output_audio:str):
    
    # Load default model
    model, df_state, _ = init_df()
    # Download and open some audio file. You use your audio files here
    audio, _ = load_audio(input_audio, sr=df_state.sr())
    # Denoise the audio
    enhanced = enhance(model, df_state, audio)

    # Save for listening
    save_audio(output_audio, enhanced, df_state.sr())
    return {"audiofile":output_audio, "message":f"Cleaned audio saved at {output_audio}"}

if __name__ == "__main__":
    def clean_audio(input_audio:str, output_path:str):
        deepfiltered_path = output_path
        vad_audio = detect_voice_activity(input_audio=input_audio, output_path=output_path, sample_rate=48000)
        print(vad_audio['message'])
        deepfiltered_audio = deepfilter(input_audio=vad_audio['audiofile'], output_audio=deepfiltered_path)
        print(deepfiltered_audio['message'])
        return deepfiltered_audio['audiofile']

    clean_audio(input_audio="D:/translation-whisper/uploaded_audios/voice1.wav", output_path="D:/translation-whisper/uploaded_audios/cleaned-voice1.wav")
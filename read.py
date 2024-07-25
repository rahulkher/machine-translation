import librosa
import scipy.fft
import noisereduce as nr
import soundfile as sf
import numpy as np

# Load the MP3 file
file_path = 'D:/translation-whisper/audios/voice1.mp3'
audio_data, sample_rate = librosa.load(file_path, sr=None)

# Apply FFT
fft_result = scipy.fft.fft(audio_data)

# Estimate the noise floor by calculating the average power of the signal
power_spectrum = np.abs(fft_result) ** 2
noise_floor = np.mean(power_spectrum) / 1  # Adjust this factor as needed

# Define a high-pass filter function to remove noise below the estimated noise floor
def dynamic_high_pass_filter(fft_data, noise_floor, sample_rate):
    frequencies = scipy.fft.fftfreq(len(fft_data), 1/sample_rate)
    filtered_fft = np.copy(fft_data)
    filtered_fft[power_spectrum < noise_floor] = 0
    return filtered_fft

# Apply dynamic high-pass filter
filtered_fft_result = dynamic_high_pass_filter(fft_result, noise_floor, sample_rate)

# Apply inverse FFT to get the time domain signal back
filtered_audio_data = scipy.fft.ifft(filtered_fft_result).real

# Apply noise reduction
# reduced_noise = nr.reduce_noise(y=filtered_audio_data, sr=sample_rate)

# Save the result to a new file
output_path = 'D:/translation-whisper/audios/cleaner-voice1.mp3'
sf.write(output_path, filtered_audio_data, sample_rate)

print(f"Original Sample Rate: {sample_rate}")
print(f"Audio Data (original): {audio_data}")
print(f"FFT Result: {fft_result}")
print(f"Filtered Audio Data: {filtered_audio_data}")
# print(f"Audio Data (noise-reduced): {reduced_noise}")

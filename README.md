
# Translation Assistance Suite

## Overview
The Translation Assistance Suite is a powerful application designed to assist in the transcription and translation of audio files, particularly focusing on translating Mandarin speech to English. The suite offers various features, including voice activity detection, noise reduction, and integration with advanced transcription and translation models.

## Features
- **Audio Upload and Processing:** Upload audio files in various formats (WAV, MP3, M4A) and process them for transcription and translation.
- **Voice Activity Detection:** Detects and trims voice activity from audio files to focus on relevant speech segments.
- **Noise Reduction:** Cleans and enhances audio quality using DeepFilterNet.
- **Transcription:** Converts Mandarin speech in audio files to text using advanced models.
- **Translation:** Translates Mandarin text to English with precise word-by-word translation.
- **User Authentication:** Secure login system with user management capabilities.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/rahulkher/machine-translation-app.git
   cd machine-translation-app
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Running the Application
1. Navigate to the project directory:
   ```sh
   cd machine-translation-app
   ```
2. Start the Streamlit app:
   ```sh
   streamlit run app.py
   ```

### Uploading and Processing Audio
- Upload audio files through the application interface.
- Select the audio file for processing.
- Use the provided options to detect voice activity, clean audio, and transcribe speech.
- Translate the transcribed text from Mandarin to English.

### Using the Translation Batch Script
1. Ensure [Ollama](https://ollama.ai) is installed and added to your system PATH.
2. Use the batch script `translate.bat` to process audio files:
   ```sh
   translate.bat <path_to_audio_file>
   ```

   This script will:
   - Check if Ollama is installed and running.
   - Activate the translation environment.
   - Run the `main.py` script to process the audio file.

## File Structure
- **app.py:** Main application script running the Streamlit app.
- **auth.py:** Handles user authentication and management.
- **deepfilternet.py:** Implements audio enhancement using DeepFilterNet.
- **main.py:** Contains the main logic for transcription and translation.
- **process_audio.py:** Provides functions for audio processing including resampling, voice activity detection, and cleaning.
- **translate.bat:** Batch script to facilitate the processing of audio files using the translation pipeline.

## Version Information
- **Version:** 1.0.0
- **Release Date:** July 25, 2024
- **Changelog:**
  - Initial release with core features including audio upload, voice activity detection, noise reduction, transcription, and translation.

## Contributing
Contributions are welcome! Please create a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License.

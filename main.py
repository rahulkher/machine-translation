import torch
import argparse
import os
import shutil
from tqdm import tqdm
import warnings
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from pathlib import Path
from process_audio import detect_voice_activity, deepfilter, reencode_audio

AUDIO_DIR = os.path.join(Path(__file__).parent, 'audios')
OUTPUT_DIR = os.path.join(Path(__file__).parent, 'outputs')

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument(
    '--audiofile',
    type=str
)
args = parser.parse_args()
# Template for creating the prompt that the AI will use to generate a response
PROMPT_TEMPLATE_RESPONSE = """
### Instructions:
You are an expert in translating mandarin to english. I will give you maindarin texts which you will translate in english.
You will pick the mandarin words one by one and give  their translations in english. I want you to give out mandarin word (in mandarin charecters)
and give its translation next. Then move to the next word. Keep on going like this till you translate all the mandarin words given to you. Dont stop in between, dont ask any permissions.  
Just print the mandarin text and english translation. Donot print anything else, just tag the mandarin word with "Mandarin:" and the translation as "Translation:"

### Mandarin text:
{question}

### Response:

"""

def transcription_model(model:str):
    # Whisper model to be used
    whisperModel = "openai/whisper-large-v3" if model=='v3' else "openai/whisper-large-v2"
    print(f"Using {whisperModel} for translation...")
    model_id = whisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)


    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device
    )
    return pipe

def translation_model(transcript, model:str, translate_prompt=PROMPT_TEMPLATE_RESPONSE):

    prompt_template = ChatPromptTemplate.from_template(translate_prompt)
    prompt = prompt_template.format(question=transcript)

    # Initialize the AI model
    model = Ollama(model=model, temperature=0.1)
    response_text = model.invoke(prompt)
    return response_text

if __name__ == "__main__":

    if str(args.audiofile) != ".":
        print(f"Transcribing and translating {args.audiofile}...")
        print()
        if not os.path.exists(os.path.join(OUTPUT_DIR, ".cache")):
            os.mkdir(os.path.join(OUTPUT_DIR, ".cache"))

        input_audio = str(args.audiofile)
        output_path = os.path.join(OUTPUT_DIR, ".cache", str(args.audiofile).replace(".mp3", ".wav"))
        deepfiltered_path = os.path.join(OUTPUT_DIR, output_path.split('\\')[-1])
    
        try:
            vad_audio = detect_voice_activity(input_audio=input_audio, output_path=output_path, sample_rate=48000)
            print(vad_audio['message'])
            print()
            deepfiltered_audio = deepfilter(input_audio=vad_audio['audiofile'], output_audio=deepfiltered_path)
            print()
            print(deepfiltered_audio['message'])
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()
            print("Audio corrupted. Repairing the audio...")
            input_audiodir = '\\'.join(input_audio.split('\\')[:-1])
            corrected_input = input_audiodir + '\corrected-' + input_audio.split('\\')[-1]
            reencode_audio(input_audio=input_audio, output_audio=corrected_input)
            vad_audio = detect_voice_activity(input_audio=corrected_input, output_path=output_path, sample_rate=48000)
            print(vad_audio['message'])
            print()
            deepfiltered_audio = deepfilter(input_audio=vad_audio['audiofile'], output_audio=deepfiltered_path)
            print()
            print(deepfiltered_audio['message'])
            print()

        print('Initiating transcription model...')
        print()
        pipe = transcription_model(model='v3')
        
        audio = deepfiltered_path
        
        if audio.endswith("wav"):
            print('Transcripting in mandarin')
            print()
            result_mandarin = pipe(audio, return_timestamps=True)
            # result_mandarin = pipe(audio, generate_kwargs={"language":"mandarin"}, return_timestamps=True)
        else:
            raise "Incorrect File type. Please pass an audio file"

        print('Translating mandarin to english')
        print()
        response_text_mandarin = translation_model(result_mandarin, model='llama3')

        print("Writing to file...")
        filename = audio.replace('.wav', '.txt')
        with open (filename, 'w', encoding='utf-8') as file:
            file.write(audio)
            file.write("\n")
            file.write('*****************************************************************************************')
            file.write("\n")
            file.write("Mandarin")
            file.write("\n")
            file.write('*****************************************************************************************')
            file.write("\n")
            file.write(result_mandarin['text'])
            file.write("\n")
            file.write('*****************************************************************************************')
            file.write("\n")
            file.write(response_text_mandarin)
        
        dir_path = os.path.join(OUTPUT_DIR, ".cache")

        try:
            shutil.rmtree(dir_path)
            print(f"Directory '{dir_path}' has been deleted successfully.")
        except Exception as e:
            print(f"Error: {e}")

    elif str(args.audiofile) == ".":
        print("Transcribing and translating all files in directory...")
        print()
        if not os.path.exists(os.path.join(OUTPUT_DIR, ".cache")):
            os.mkdir(os.path.join(OUTPUT_DIR, ".cache"))

        audios = os.listdir(AUDIO_DIR)

        for audio in tqdm(audios, desc="Processing audios", unit="audio"):

            input_audio = os.path.join(AUDIO_DIR, audio)
            output_path = os.path.join(OUTPUT_DIR, ".cache", audio.replace(".mp3", ".wav"))
            deepfiltered_path = os.path.join(OUTPUT_DIR, output_path.split('\\')[-1])
        
            try:
                vad_audio = detect_voice_activity(input_audio=input_audio, output_path=output_path, sample_rate=48000)
                print(vad_audio['message'])
                print()
                deepfiltered_audio = deepfilter(input_audio=vad_audio['audiofile'], output_audio=deepfiltered_path)
                print()
                print(deepfiltered_audio['message'])
                print()
            except Exception as e:
                print(f"Error: {e}")
                print()
                print("Audio corrupted. Repairing the audio...")
                input_audiodir = '\\'.join(output_path.split('\\')[:-1])
                corrected_input = input_audiodir + '\corrected-' + input_audio.split('\\')[-1]
                reencode_audio(input_audio=input_audio, output_audio=corrected_input)
                vad_audio = detect_voice_activity(input_audio=corrected_input, output_path=output_path, sample_rate=48000)
                print(vad_audio['message'])
                print()
                deepfiltered_audio = deepfilter(input_audio=vad_audio['audiofile'], output_audio=deepfiltered_path)
                print()
                print(deepfiltered_audio['message'])
                print()

            print('Initiating transcription model...')
            print()
            pipe = transcription_model(model='v3')
            
            audio = deepfiltered_path
            
            if audio.endswith("wav"):
                print('Transcripting in mandarin')
                print()
                result_mandarin = pipe(audio, return_timestamps=True)
                # result_mandarin = pipe(audio, generate_kwargs={"language":"mandarin"}, return_timestamps=True)
            else:
                raise "Incorrect File type. Please pass an audio file"

            print('Translating mandarin to english')
            print()
            response_text_mandarin = translation_model(result_mandarin, model='llama3')

            print("Writing to file...")
            filename = audio.replace('.wav', '.txt')
            with open (filename, 'w', encoding='utf-8') as file:
                file.write(audio)
                file.write("\n")
                file.write('*****************************************************************************************')
                file.write("\n")
                file.write("Mandarin")
                file.write("\n")
                file.write('*****************************************************************************************')
                file.write("\n")
                file.write(result_mandarin['text'])
                file.write("\n")
                file.write('*****************************************************************************************')
                file.write("\n")
                file.write(response_text_mandarin)

            dir_path = os.path.join(OUTPUT_DIR, ".cache")

            try:
                shutil.rmtree(dir_path)
                print(f"Directory '{dir_path}' has been deleted successfully.")
            except Exception as e:
                print(f"Error: {e}")




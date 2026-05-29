# Install required libraries
# !pip install -q edge-tts gradio nest_asyncio pydub

import os
import io
import re
import time
import asyncio
import nest_asyncio
import edge_tts
import gradio as gr
from pydub import AudioSegment

# Enable nested asyncio
nest_asyncio.apply()

# Free up background ports
gr.close_all()

# Global Voice Mapping Objects
ALL_VOICES_MAP = {}
INDIAN_VOICES = {}
MULTILINGUAL_VOICES = {}
OTHER_VOICES = {}

async def load_and_categorize_voices():
    global ALL_VOICES_MAP, INDIAN_VOICES, MULTILINGUAL_VOICES, OTHER_VOICES
    try:
        voices = await edge_tts.list_voices()
        ALL_VOICES_MAP.clear()
        INDIAN_VOICES.clear()
        MULTILINGUAL_VOICES.clear()
        OTHER_VOICES.clear()

        for v in voices:
            locale = v['Locale']
            short_name = v['ShortName']
            gender = v['Gender']
            display_name = f"{locale} | {short_name} ({gender})"
            
            ALL_VOICES_MAP[display_name] = short_name
            
            is_indian = (
                locale.endswith("-IN") or 
                locale.startswith("hi-") or 
                locale.startswith("bn-") or 
                locale.startswith("ta-") or 
                locale.startswith("te-") or 
                locale.startswith("mr-") or 
                locale.startswith("gu-") or 
                locale.startswith("kn-") or 
                locale.startswith("ml-") or 
                locale.startswith("ur-")
            )
            
            if is_indian:
                INDIAN_VOICES[display_name] = short_name
            elif "multilingual" in short_name.lower():
                MULTILINGUAL_VOICES[display_name] = short_name
            else:
                OTHER_VOICES[display_name] = short_name
                
        INDIAN_VOICES = dict(sorted(INDIAN_VOICES.items()))
        MULTILINGUAL_VOICES = dict(sorted(MULTILINGUAL_VOICES.items()))
        OTHER_VOICES = dict(sorted(OTHER_VOICES.items()))

        print(f"Loaded: {len(INDIAN_VOICES)} Indian, {len(MULTILINGUAL_VOICES)} Multilingual, {len(OTHER_VOICES)} Other voices.")
    except Exception as e:
        print(f"Error loading voices at startup: {str(e)}")

# Safe Text cleaner
def clean_text_for_tts(text):
    if not text:
        return ""
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Master high quality 16-bit CD exporter
def export_processed_audio(audio_segment, file_name, audio_format, bitrate, sample_rate):
    try:
        if audio_segment is None or len(audio_segment) == 0:
            return None
        audio_segment = audio_segment.set_sample_width(2)
        target_hz = int(sample_rate.replace("Hz", ""))
        audio_segment = audio_segment.set_frame_rate(target_hz)
        
        export_kwargs = {}
        if audio_format in ["mp3", "m4a", "ogg"]:
            export_kwargs["bitrate"] = bitrate
            
        output_file = f"{file_name}.{audio_format}"
        audio_segment.export(output_file, format=audio_format, **export_kwargs)
        return output_file
    except Exception as e:
        print(f"Export Error: {e}")
        return None

# Simple Text to Speech Pipeline
async def process_simple_tts(text, voice_selection, rate, pitch, audio_format, bitrate, sample_rate, progress=gr.Progress()):
    if not text or voice_selection not in ALL_VOICES_MAP:
        return "Please fill all options.", None
    try:
        progress(0.0, desc="Connecting to Edge API...")
        voice_name = ALL_VOICES_MAP[voice_selection]
        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}%" if pitch == 0 else f"{pitch:+d}Hz"
        
        cleaned_text = clean_text_for_tts(text)
        communicate = edge_tts.Communicate(cleaned_text, voice_name, rate=rate_str, pitch=pitch_str)
        
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        audio_buffer.seek(0)
        
        raw_audio = AudioSegment.from_file(audio_buffer, format="mp3")
        final_file = export_processed_audio(raw_audio, "tts_output", audio_format, bitrate, sample_rate)
        return "Success!", final_file
    except Exception as e:
        return f"Error: {e}", None

# Run voice loader at startup safely
try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(load_and_categorize_voices())
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(load_and_categorize_voices())

# --- VIBRANT GREEN UI DESIGN ---
green_theme = gr.themes.Default(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="slate"
)

# UI Dashboard Definitions
with gr.Blocks(title="Studio Quality TTS Studio", theme=green_theme) as demo:
    gr.Markdown("# 🎙️ Professional Studio Quality Text-to-Speech")
    gr.Markdown("> 💡 **Fast Download Tip:** Audio compile hone ke baad instantly download ke liye **MP3** aur Bitrate ko **192k** select karein.")

    def update_voices_by_category(category):
        if category == "Indian Voices 🇮🇳":
            choices = list(INDIAN_VOICES.keys())
        elif category == "Multilingual Voices 🌐":
            choices = list(MULTILINGUAL_VOICES.keys())
        else:
            choices = list(OTHER_VOICES.keys())
        default_val = choices[0] if choices else None
        return gr.Dropdown(choices=choices, value=default_val)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Text to Speech", placeholder="Type here...", lines=5)
            
            voice_category_tts = gr.Radio(
                ["Indian Voices 🇮🇳", "Multilingual Voices 🌐", "Other Voices 🌍"], 
                label="Voice Category", 
                value="Indian Voices 🇮🇳"
            )
            voice_dropdown_tts = gr.Dropdown(
                choices=list(INDIAN_VOICES.keys()), 
                label="Select Voice", 
                value=list(INDIAN_VOICES.keys())[0] if INDIAN_VOICES else None, 
                filterable=True
            )

            with gr.Row():
                rate_slider_tts = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Speed (%)")
                pitch_slider_tts = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Pitch (Hz)")

            with gr.Group():
                gr.Markdown("### 🎚️ Studio Export Settings")
                export_format_tts = gr.Dropdown(["mp3", "wav", "flac", "ogg", "m4a"], label="Format", value="mp3")
                export_bitrate_tts = gr.Dropdown(["320k", "256k", "192k", "128k", "64k"], label="Bitrate", value="320k")
                export_hz_tts = gr.Dropdown(["96000Hz", "48000Hz", "44100Hz", "24000Hz"], label="Sample Rate (Hz)", value="48000Hz")

            submit_btn_tts = gr.Button("Generate Audio", variant="primary")

        with gr.Column():
            status_msg_tts = gr.Textbox(label="Status", interactive=False)
            audio_output_tts = gr.Audio(label="Audio Output", type="filepath")

    voice_category_tts.change(fn=update_voices_by_category, inputs=voice_category_tts, outputs=voice_dropdown_tts)
    
    submit_btn_tts.click(
        fn=process_simple_tts,
        inputs=[
            input_text, voice_dropdown_tts, rate_slider_tts, pitch_slider_tts, 
            export_format_tts, export_bitrate_tts, export_hz_tts
        ],
        outputs=[status_msg_tts, audio_output_tts]
    )

# Launch Dashboard
demo.launch(share=True, debug=True, max_threads=100)

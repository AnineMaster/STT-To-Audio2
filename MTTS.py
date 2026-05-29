# Install required libraries
# !pip install -q edge-tts pydub nest-asyncio pandas matplotlib

import os
import io
import re
import time
import asyncio
import nest_asyncio
import pandas as pd
import edge_tts
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

# Dynamic Voice Loader & Categorizer
async def get_categorized_voices():
    global ALL_VOICES_MAP, INDIAN_VOICES, MULTILINGUAL_VOICES, OTHER_VOICES
    try:
        all_v = await edge_tts.list_voices()
        total_count = len(all_v)
        multi, indian, other = [], [], []
        indian_locales = ['hi-IN', 'en-IN', 'bn-IN', 'gu-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'ta-IN', 'te-IN', 'ur-IN']
        
        ALL_VOICES_MAP.clear()
        INDIAN_VOICES.clear()
        MULTILINGUAL_VOICES.clear()
        OTHER_VOICES.clear()

        for v in all_v:
            label, value = f"{v['FriendlyName']} ({v['Locale']})", v['ShortName']
            ALL_VOICES_MAP[label] = value
            ALL_VOICES_MAP[value] = value
            
            if "Multilingual" in v['FriendlyName']: 
                multi.append((label, value))
                MULTILINGUAL_VOICES[label] = value
            elif any(loc in v['Locale'] for loc in indian_locales): 
                indian.append((label, value))
                INDIAN_VOICES[label] = value
            else: 
                other.append((label, value))
                OTHER_VOICES[label] = value

        INDIAN_VOICES = dict(sorted(INDIAN_VOICES.items()))
        MULTILINGUAL_VOICES = dict(sorted(MULTILINGUAL_VOICES.items()))
        OTHER_VOICES = dict(sorted(OTHER_VOICES.items()))
        
        return sorted(multi), sorted(indian), sorted(other), total_count
    except Exception as e:
        print(f"Error loading voices: {e}")
        return [], [], [], 0

# Safe Text cleaner (Ignores Brackets content and minus/hyphen symbols)
def clean_text_for_tts(text):
    if not text:
        return ""
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = text.replace('-', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def time_to_ms(time_str):
    try:
        time_str = time_str.replace(',', '.')
        h, m, s = time_str.split(':')
        return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)
    except:
        return 0

def stretch_audio(audio, target_duration_ms):
    if len(audio) == 0 or target_duration_ms <= 0:
        return audio
    speed_ratio = len(audio) / target_duration_ms
    if speed_ratio > 1.1:
        applied_speed = min(speed_ratio, 2.0)
        return audio.speedup(playback_speed=applied_speed, chunk_size=50, crossfade=25)
    return audio

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

# Safe Document Log Exporter (TXT, CSV, PDF)
def save_log_formats(log_df):
    try:
        if log_df is None or log_df.empty:
            return None, None, None
        
        csv_path = "generation_log.csv"
        log_df.to_csv(csv_path, index=False)
        
        txt_path = "generation_log.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(log_df.to_string(index=False))
            
        pdf_path = "generation_log.pdf"
        fig, ax = plt.subplots(figsize=(7, len(log_df) * 0.35 + 1.5))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=log_df.values, 
            colLabels=log_df.columns, 
            loc='center', 
            cellLoc='center',
            colColours=["#2ecc71"] * len(log_df.columns)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.3)
        
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        return txt_path, csv_path, pdf_path
    except Exception as e:
        print(f"Log Export Error: {e}")
        return None, None, None

# parse SRT to unlimited speaker grid automatically
def parse_srt_to_unlimited_speaker_grid(srt_content):
    srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
    rows = []
    blocks = re.split(r'\n\s*\n', srt_content.strip())

    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
        try:
            timing_line_index = -1
            for i, line in enumerate(lines):
                if '-->' in line:
                    timing_line_index = i
                    break

            if timing_line_index == -1:
                continue

            time_str = lines[timing_line_index]
            times = [t.strip() for t in time_str.split('-->')]
            start_time_str = times[0]
            end_time_str = times[1]

            text_lines = lines[timing_line_index + 1:]
            full_text = " ".join([line.strip() for line in text_lines if line.strip()])

            speaker = "Speaker 1"
            text_content = full_text
            
            match_bracket = re.match(r'^\[\s*(.*?)\s*\]\s*(.*)', full_text)
            match_colon = re.match(r'^(.*?)\s*:\s*(.*)', full_text)
            
            if match_bracket:
                speaker = match_bracket.group(1).strip()
                text_content = match_bracket.group(2).strip()
            elif match_colon:
                if not re.match(r'^\d{2}$', match_colon.group(1)):
                    speaker = match_colon.group(1).strip()
                    text_content = match_colon.group(2).strip()

            rows.append([lines[0] if timing_line_index > 0 else "?", start_time_str, end_time_str, speaker, text_content])
        except Exception as e:
            print(f"Warning parsing block: {e}")
            continue
    if not rows:
        return [["1", "00:00:00,000", "00:00:03,000", "Speaker 1", "Sample text here"]]
    return rows

# Convert Multi-Speaker visual Dataframe back to segments (Unlimited Speakers)
def df_to_unlimited_speaker_segments(df_data):
    segments = []
    records = df_data.values.tolist() if isinstance(df_data, pd.DataFrame) else df_data

    for row in records:
        if len(row) < 5: continue
        try:
            start_ms = time_to_ms(str(row[1]).strip())
            end_ms = time_to_ms(str(row[2]).strip())
            speaker = str(row[3]).strip()
            text = clean_text_for_tts(str(row[4]).strip())
            if text:
                segments.append({
                    'start': start_ms, 
                    'end': end_ms, 
                    'speaker': speaker,
                    'text': text
                })
        except Exception as e:
            print(f"Error parsing row {row}: {e}")
    return segments

# Unlimited Multi-Speaker SRT to Audio Pipeline
async def process_unlimited_speaker_srt(grid_data, speaker_mapping, speed_val, pitch_val, audio_format, bitrate, sample_rate, progress=gr.Progress()):
    segments = df_to_unlimited_speaker_segments(grid_data)
    if not segments:
        yield "No valid subtitle rows were extracted.", None, None, None, None, None, None
        return

    start_process_time = time.time()
    total_segments = len(segments)
    
    # AI Estimate Engine
    total_chars = sum(len(d['text']) for d in segments)
    estimated_compile_time_s = round((total_segments * 0.30) + (total_chars * 0.001), 1)
    
    yield f"🔄 AI Estimate: Total Segments: {total_segments} | Estimated Process Time: ~{estimated_compile_time_s}s. Starting...", None, None, None, None, None, None
    await asyncio.sleep(0.8)

    # Convert mapping table back to dictionary map
    spk_records = speaker_mapping.values.tolist() if isinstance(speaker_mapping, pd.DataFrame) else speaker_mapping
    voice_map = {}
    for r in spk_records:
        if len(r) >= 2:
            voice_map[str(r[0]).strip()] = ALL_VOICES_MAP.get(str(r[1]).strip(), str(r[1]).strip())

    combined_audio = AudioSegment.silent(duration=0)
    actual_spoken_ms = 0
    gen_times = []
    rate_str = f"{speed_val:+d}%"
    pitch_str = f"{pitch_val:+d}%" if pitch_val == 0 else f"{pitch_val:+d}Hz"

    for i, s in enumerate(segments):
        current_msg = f"Processing segment {i+1} of {total_segments}... [Est. Time Remaining: {round(max(0, estimated_compile_time_s - (i * 0.35)), 1)}s]"
        yield current_msg, None, None, None, None, None, None

        start_seg_time = time.time()
        temp_file = f"multi_temp_{i}.mp3"
        v_name = voice_map.get(s['speaker'], "hi-IN-SwaraNeural")
        
        cleaned_text = clean_text_for_tts(s['text'])
        communicate = edge_tts.Communicate(cleaned_text, v_name, rate=rate_str, pitch=pitch_str)
        await communicate.save(temp_file)

        if os.path.exists(temp_file):
            seg_audio = AudioSegment.from_file(temp_file, format="mp3")
            target_start = s['start']
            target_end = s['end']
            allowed_duration = target_end - target_start

            if allowed_duration > 0 and len(seg_audio) > allowed_duration:
                seg_audio = stretch_audio(seg_audio, allowed_duration)

            actual_spoken_ms += len(seg_audio)
            current_pos = len(combined_audio)
            if current_pos < target_start:
                combined_audio += AudioSegment.silent(duration=target_start - current_pos)

            combined_audio += seg_audio
            os.remove(temp_file)
            
        gen_times.append(round(time.time() - start_seg_time, 3))

    # Export using High-Quality CD Master
    final_file = export_processed_audio(combined_audio, "final_multi_srt_audio", audio_format, bitrate, sample_rate)
    total_timeline_sec = round(len(combined_audio) / 1000.0, 2)
    spoken_only_sec = round(actual_spoken_ms / 1000.0, 2)
    
    elapsed_generation = round(time.time() - start_process_time, 2)
    avg_time = round(elapsed_generation / total_segments, 2) if total_segments > 0 else 0

    final_status = (f"✅ Done! Audio Duration: {total_timeline_sec}s\n"
                    f"Generated in {elapsed_generation}s (Avg: {avg_time}s/seg).")

    log_df = pd.DataFrame({
        "Segment Index": range(1, total_segments + 1),
        "Speaker": [s['speaker'] for s in segments],
        "Target Duration (s)": [round((s['end']-s['start'])/1000.0, 2) for s in segments],
        "Generation Time (s)": gen_times
    })

    txt_f, csv_f, pdf_f = save_log_formats(log_df)

    yield final_status, final_file, final_file, log_df, txt_f, csv_f, pdf_f

# Parse SRT and auto-generate unique speaker labels for mapping table
def load_srt_and_auto_extract_speakers(t, f, txt):
    grid_data = parse_srt_to_unlimited_speaker_grid(open(f.name).read() if t == "Upload File" and f else txt)
    unique_speakers = sorted(list(set(row[3] for row in grid_data)))
    
    mapping_data = []
    all_voice_keys = list(ALL_VOICES_MAP.keys())
    
    for i, spk in enumerate(unique_speakers):
        default_voice = all_voice_keys[0]
        for vk in all_voice_keys:
            if "Swara" in vk and i == 0:
                default_voice = vk
                break
            elif "Madhur" in vk and i == 1:
                default_voice = vk
                break
            elif "Neerja" in vk and i == 2:
                default_voice = vk
                break
                
        mapping_data.append([spk, default_voice])
        
    return grid_data, mapping_data

# Run voice loader inside nested-asyncio context safely
try:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_categorized_voices())
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(get_categorized_voices())

# --- VIBRANT GREEN UI DESIGN ---
green_theme = gr.themes.Default(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="slate"
)

# UI Dashboard Definitions
with gr.Blocks(title="Studio Quality SRT Voiceover Editor", theme=green_theme) as demo:
    gr.Markdown("# 🎙️ Unlimited Multi-Speaker Studio Editor")
    gr.Markdown("> 💡 **Fast Download Tip:** Audio compile hone ke baad instantly download ke liye **MP3** aur Bitrate ko **192k** select karein.")

    with gr.Row():
        with gr.Column(scale=1):
            m_in_type = gr.Radio(["Upload File", "Paste Text"], value="Upload File", label="SRT Input Method")
            m_file = gr.File(label="Upload SRT File", file_types=[".srt"])
            m_text = gr.Textbox(label="Paste SRT Text", placeholder="Paste SRT content here...", lines=5, visible=False)
            load_m = gr.Button("📂 Load Subtitles & Extract Speakers", variant="secondary")
            
            # Dynamic Speaker Mapping Table (No limits - unlimited speakers!)
            gr.Markdown("### 👤 Speaker Voice Mapping")
            mapping_grid = gr.Dataframe(
                headers=["Speaker Label", "Assigned Voice (search keys above)"],
                datatype=["str", "str"],
                col_count=(2, "fixed"),
                interactive=True,
                value=[["Speaker 1", "hi-IN-SwaraNeural"]]
            )

            with gr.Row():
                rate_multi_slider = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Default Speed (%)")
                pitch_multi_slider = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Pitch (Hz)")

            with gr.Group():
                gr.Markdown("### 🎚️ Studio Export Settings")
                multi_export_format = gr.Dropdown(["mp3", "wav", "flac", "ogg", "m4a"], label="Format", value="mp3")
                multi_export_bitrate = gr.Dropdown(["320k", "256k", "192k", "128k", "64k"], label="Bitrate", value="320k")
                multi_export_hz = gr.Dropdown(["96000Hz", "48000Hz", "44100Hz", "24000Hz"], label="Sample Rate (Hz)", value="48000Hz")

            submit_btn_multi = gr.Button("⚡ Generate Multi-Speaker Audio", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 📝 Interactive Subtitle Editor (Type custom speaker name/label in the Speaker column)")
            
            # Visual Editable Grid
            editor_grid = gr.Dataframe(
                headers=["Index", "Start Time", "End Time", "Speaker Label", "Subtitle Text"],
                datatype=["str", "str", "str", "str", "str"],
                col_count=(5, "fixed"),
                interactive=True,
                wrap=True,
                value=[["1", "00:00:00,000", "00:00:03,000", "Speaker 1", "Upload a file or paste text, then click Load."]]
            )
            
            status_msg_multi = gr.Textbox(label="Status", interactive=False)
            audio_output_multi = gr.Audio(label="Preview Audio", type="filepath")
            file_multi_dl = gr.File(label="Download Audio")
            
            # Dynamic Download Files Row (TXT, CSV, PDF)
            with gr.Row():
                multi_txt_dl = gr.File(label="Download TXT Log")
                multi_csv_dl = gr.File(label="Download CSV Log")
                multi_pdf_dl = gr.File(label="Download PDF Log")
            
            log_multi_grid = gr.Dataframe(label="Google Sheet Log (Target vs Gen Time)")

    # Visual Grid input toggle
    m_in_type.change(lambda v: (gr.update(visible=v=="Upload File"), gr.update(visible=v=="Paste Text")), inputs=m_in_type, outputs=[m_file, m_text])

    # Load raw data and dynamically generate speaker mapping rows on load click
    load_m.click(
        fn=load_srt_and_auto_extract_speakers,
        inputs=[m_in_type, m_file, m_text],
        outputs=[editor_grid, mapping_grid]
    )

    # Process multi-speaker compilation
    submit_btn_multi.click(
        fn=process_unlimited_speaker_srt,
        inputs=[
            editor_grid, mapping_grid,
            rate_multi_slider, pitch_multi_slider, multi_export_format, multi_export_bitrate, multi_export_hz
        ],
        outputs=[status_msg_multi, audio_output_multi, file_multi_dl, log_multi_grid, multi_txt_dl, multi_csv_dl, multi_pdf_dl]
    )

# Launch Dashboard
demo.launch(share=True, debug=True, max_threads=100)

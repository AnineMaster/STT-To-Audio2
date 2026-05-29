import os
import io
import re
import time
import pandas as pd
import asyncio
import nest_asyncio
import edge_tts
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pydub import AudioSegment

# Enable notebook loop safety
nest_asyncio.apply()

# Free up background zombie ports before starting
gr.close_all()

# Global Voice Mapping Objects
ALL_VOICES_MAP = {}
INDIAN_VOICES = {}
MULTILINGUAL_VOICES = {}
OTHER_VOICES = {}

# Dynamic Voice Loader & Categorizer [100% Fixed and Cleaned]
async def get_categorized_voices():
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
            ALL_VOICES_MAP[short_name] = short_name
            
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
                
        # Sort collections
        INDIAN_VOICES = dict(sorted(INDIAN_VOICES.items()))
        MULTILINGUAL_VOICES = dict(sorted(MULTILINGUAL_VOICES.items()))
        OTHER_VOICES = dict(sorted(OTHER_VOICES.items()))

        print(f"Loaded: {len(INDIAN_VOICES)} Indian, {len(MULTILINGUAL_VOICES)} Multilingual, {len(OTHER_VOICES)} Other voices.")
    except Exception as e:
        print(f"Error loading voices at startup: {str(e)}")

# Safe Text cleaner (Ignores Brackets content and minus symbols)
def clean_text_for_tts(text):
    if not text:
        return ""
    # Remove text inside round brackets (), square brackets [], and curly brackets {}
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    # Replace minus/hyphens with a space to prevent pronunciation errors
    text = text.replace('-', ' ')
    # Trim and normalize double spaces
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
        
        # Enforce 16-Bit Depth (CD Studio Mastering Standard)
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
        
        # Custom elegant green table layout
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

# Parse SRT content safely
def parse_srt_with_timings(srt_content):
    srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
    segments = []
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
            start_ms = time_to_ms(times[0])
            end_ms = time_to_ms(times[1])

            text_lines = lines[timing_line_index + 1:]
            text = " ".join([line.strip() for line in text_lines if line.strip()])

            if text:
                segments.append({'start': start_ms, 'end': end_ms, 'text': text})
        except Exception as e:
            print(f"Warning parsing block: {e}")
            continue
    return segments

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

            # Auto-detect speakers, default to "Speaker 1"
            speaker = "Speaker 1"
            text_content = full_text
            
            match_bracket = re.match(r'^\[\s*(.*?)\s*\]\s*(.*)', full_text)
            match_colon = re.match(r'^(.*?)\s*:\s*(.*)', full_text)
            
            if match_bracket:
                speaker = match_bracket.group(1).strip()
                text_content = match_bracket.group(2).strip()
            elif match_colon:
                # Avoid matching raw timestamps (e.g. HH:MM)
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

# In-memory single chunk generator helper (Zero Disk Writes)
async def fetch_tts_segment_in_memory(sem, text, voice_name, rate_str, pitch_str, idx, log_list=None):
    async with sem:
        start_time = time.time()
        try:
            cleaned_text = clean_text_for_tts(text)
            if not cleaned_text:
                return AudioSegment.silent(duration=100), idx

            communicate = edge_tts.Communicate(cleaned_text, voice_name, rate=rate_str, pitch=pitch_str)
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            audio_buffer.seek(0)
            
            segment_audio = AudioSegment.from_file(audio_buffer, format="mp3")
            gen_time = time.time() - start_time
            if log_list is not None:
                log_list[idx] = gen_time
            return segment_audio, idx
        except Exception as e:
            print(f"Fetch failed for segment: {e}")
            return None, idx

# 1. Single Speaker SRT to Audio Pipeline (with Grid Log & Safe clean) [100% Fixed Parameters Alignment]
async def process_edge_srt(srt_type, srt_file, srt_text, voice_selection, speed_val, pitch_val, audio_format, bitrate, sample_rate, progress=gr.Progress()):
    if voice_selection not in ALL_VOICES_MAP:
        yield None, None, "Please select a valid voice.", None, None, None, None
        return

    srt_content = ""
    if srt_type == "Upload File" and srt_file:
        try:
            with open(srt_file.name, 'r', encoding='utf-8') as f:
                srt_content = f.read()
        except Exception as e:
            yield None, None, f"Error reading file: {e}", None, None, None, None
            return
    elif srt_type == "Paste Text" and srt_text:
        srt_content = srt_text

    if not srt_content.strip():
        yield None, None, "No SRT content provided.", None, None, None, None
        return

    voice = ALL_VOICES_MAP[voice_selection]
    start_process_time = time.time()
    segments = parse_srt_with_timings(srt_content)
    combined_audio = AudioSegment.silent(duration=0)
    actual_spoken_ms = 0
    total_segments = len(segments)

    if total_segments == 0:
        yield None, None, "No valid segments found in SRT.", None, None, None, None
        return

    # AI-like Instant Duration Estimation Engine
    total_chars = sum(len(d['text']) for d in segments)
    estimated_compile_time_s = round((total_segments * 0.30) + (total_chars * 0.001), 1)
    
    est_msg = f"🔄 AI Estimate: Total Segments: {total_segments} | Estimated Process Time: ~{estimated_compile_time_s}s. Starting..."
    yield None, None, est_msg, None, None, None, None
    await asyncio.sleep(0.8)

    gen_times = [0.0] * total_segments
    rate_str = f"{speed_val:+d}%"
    pitch_str = f"{pitch_val:+d}%" if pitch_val == 0 else f"{pitch_val:+d}Hz"

    # --- PASS 1: Parallel Estimations ---
    progress(0.0, desc="Pass 1: Estimating segment times...")
    est_audios = [None] * total_segments
    est_tasks = [fetch_tts_segment_in_memory(asyncio.Semaphore(15), s['text'], voice, rate_str, pitch_str, i) for i, s in enumerate(segments)]
    
    completed_est = 0
    for future in asyncio.as_completed(est_tasks):
        audio_segment, idx = await future
        est_audios[idx] = audio_segment
        completed_est += 1
        progress(completed_est / total_segments, desc=f"Pass 1: Estimating ({completed_est}/{total_segments})")

    est_track_duration_s = sum([len(a)/1000 for a in est_audios if a]) / (1 + (speed_val/100))

    # --- PASS 2: Squeeze-to-Fit Generation ---
    progress(0.35, desc=f"Track Est: {est_track_duration_s:.1f}s | Pass 2: Generating...")
    final_audios = [None] * total_segments
    final_tasks = []
    for i, seg in enumerate(segments):
        target = max(1, seg['end'] - seg['start'])
        if est_audios[i] is None: continue
        
        multiplier = len(est_audios[i]) / target
        new_rate = f"{max(-35, min(75, ((1 + (speed_val/100)) * multiplier - 1) * 100)):+.0f}%"
        final_tasks.append(fetch_tts_segment_in_memory(asyncio.Semaphore(15), seg['text'], voice, new_rate, pitch_str, i, gen_times))

    pass2_start = time.time()
    completed_final = 0
    for future in asyncio.as_completed(final_tasks):
        audio_segment, idx = await future
        final_audios[idx] = audio_segment
        completed_final += 1
        progress(completed_final / total_segments, desc=f"Track Est: {est_track_duration_s:.1f}s | Pass 2: Generating ({completed_final}/{total_segments})")
    
    pass2_duration = time.time() - pass2_start

    # --- PASS 3: Absolute Canvas Stitching (0ms Cumulative Drift) ---
    progress(0.8, desc="Pass 3: Stitching absolute timeline...")
    total_canvas_duration = segments[-1]['end'] + 10000
    combined_audio = AudioSegment.silent(duration=total_canvas_duration)

    for i, segment in enumerate(segments):
        start_ms = segment['start']
        end_ms = segment['end']
        target_srt_slot_duration = max(1, end_ms - start_ms)
        speech_audio = final_audios[i]
        if speech_audio is None: continue

        # Strict Overlap Guard: If speech overflows, apply soft fade-out trim
        if len(speech_audio) > target_srt_slot_duration:
            fade_len = min(100, target_srt_slot_duration // 2)
            speech_audio = speech_audio[:target_srt_slot_duration].fade_out(fade_len)

        # Overlay onto exact absolute millisecond position
        combined_audio = combined_audio.overlay(speech_audio, position=start_ms)
        progress(0.8 + (i/total_segments)*0.15, desc="Pass 3: Stitching absolute timeline...")

    # Crop canvas exactly at the real absolute end
    combined_audio = combined_audio[:segments[-1]['end']]

    # Export using High-Quality CD Master
    progress(0.95, desc="Exporting master audio...")
    final_file = export_processed_audio(combined_audio, "final_srt_audio", audio_format, bitrate, sample_rate)
    total_duration = time.time() - start_process_time
    
    # Sateek Wall-Clock Average Segment Generation speed
    avg_segment_time = pass2_duration / total_segments if total_segments else 0

    final_status = (f"✅ Done! Audio Duration: {est_track_duration_s:.2f}s | "
                    f"Total Compile Time: {total_duration:.2f}s | "
                    f"Average Segment Speed: {avg_segment_time:.3f}s")

    # Generate Google Sheets-like Grid Data log
    log_df = pd.DataFrame({
        "Segment Index": range(1, total_segments + 1),
        "Target Duration (s)": [round((s['end']-s['start'])/1000.0, 2) for s in segments],
        "Generation Time (s)": gen_times
    })

    txt_f, csv_f, pdf_f = save_log_formats(log_df)
    yield final_file, final_file, final_status, log_df, txt_f, csv_f, pdf_f

# 2. Unlimited Multi-Speaker SRT to Audio Pipeline
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

    rate_str = f"{speed_val:+d}%"
    pitch_str = f"{pitch_val:+d}%" if pitch_val == 0 else f"{pitch_val:+d}Hz"
    gen_times = [0.0] * total_segments

    # --- PASS 1: Parallel Estimations ---
    progress(0.0, desc="Pass 1: Estimating segment times...")
    est_audios = [None] * total_segments
    est_tasks = []
    for i, seg in enumerate(segments):
        v_name = voice_map.get(seg['speaker'], "hi-IN-SwaraNeural")
        est_tasks.append(fetch_tts_segment_in_memory(asyncio.Semaphore(15), seg['text'], v_name, f"{speed_val:+.0f}%", pitch_str, i))
        
    completed_est = 0
    for future in asyncio.as_completed(est_tasks):
        audio_segment, idx = await future
        est_audios[idx] = audio_segment
        completed_est += 1
        progress(completed_est / total_segments, desc=f"Pass 1: Estimating ({completed_est}/{total_segments})")

    est_track_duration_s = sum([len(a)/1000 for a in est_audios if a]) / (1 + (speed_val/100))

    # --- PASS 2: Squeeze-to-Fit Generation ---
    progress(0.35, desc=f"Track Est: {est_track_duration_s:.1f}s | Pass 2: Generating...")
    final_audios = [None] * total_segments
    final_tasks = []
    for i, s in enumerate(segments):
        target = max(1, s['end'] - s['start'])
        if est_audios[i] is None: continue
        mult = len(est_audios[i]) / target
        nr = f"{max(-35, min(75, ((1 + (speed_val/100)) * mult - 1) * 100)):+.0f}%"
        v_name = voice_map.get(s['speaker'], "hi-IN-SwaraNeural")
        final_tasks.append(fetch_tts_segment_in_memory(asyncio.Semaphore(15), s['text'], v_name, nr, pitch_str, i, gen_times))
        
    pass2_start = time.time()
    completed_final = 0
    for future in asyncio.as_completed(final_tasks):
        audio_segment, idx = await future
        final_audios[idx] = audio_segment
        completed_final += 1
        progress(completed_final / total_segments, desc=f"Track Est: {est_track_duration_s:.1f}s | Pass 2: Generating ({completed_final}/{total_segments})")
        
    pass2_duration = time.time() - pass2_start

    # --- PASS 3: Absolute Canvas Stitching (0ms Cumulative Drift) ---
    progress(0.8, desc="Pass 3: Stitching absolute timeline...")
    total_canvas_duration = segments[-1]['end'] + 10000
    combined_audio = AudioSegment.silent(duration=total_canvas_duration)

    for i, s in enumerate(segments):
        start_ms = s['start']
        end_ms = s['end']
        target_slot_duration = max(1, end_ms - start_ms)
        speech_audio = final_audios[i]
        if speech_audio is None: continue

        # Strict Overlap Guard: If speech overflows, apply soft fade-out trim
        if len(speech_audio) > target_slot_duration:
            fade_len = min(100, target_slot_duration // 2)
            speech_audio = speech_audio[:target_slot_duration].fade_out(fade_len)

        # Overlay onto exact absolute millisecond position
        combined_audio = combined_audio.overlay(speech_audio, position=start_ms)
        progress(0.8 + (i/total_segments)*0.15, desc="Pass 3: Stitching absolute timeline...")

    # Crop canvas exactly at the real absolute end
    combined_audio = combined_audio[:segments[-1]['end']]

    # Export using High-Quality CD Master
    progress(0.95, desc="Exporting master audio...")
    final_file = export_processed_audio(combined_audio, "final_multi_srt_audio", audio_format, bitrate, sample_rate)
    total_duration = time.time() - start_process_time
    
    # Sateek Wall-Clock Average Segment Generation speed
    avg_segment_time = pass2_duration / total_segments if total_segments else 0

    final_status = (f"✅ Done! Audio Duration: {est_track_duration_s:.2f}s | "
                    f"Total Compile Time: {total_duration:.2f}s | "
                    f"Average Segment Speed: {avg_segment_time:.3f}s")

    log_df = pd.DataFrame({
        "Segment Index": range(1, total_segments + 1),
        "Speaker": [s['speaker'] for s in segments],
        "Target Duration (s)": [round((s['end']-s['start'])/1000.0, 2) for s in segments],
        "Generation Time (s)": gen_times
    })

    # Generate log downloads (TXT, CSV, PDF)
    txt_f, csv_f, pdf_f = save_log_formats(log_df)

    yield final_status, final_file, final_file, log_df, txt_f, csv_f, pdf_f

# 3. Simple Text to Speech Pipeline
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
    gr.Markdown("# 🎙️ Professional Studio Quality Voiceover Editor")
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

    with gr.Tab("SRT to Audio (Single Speaker)"):
        with gr.Row():
            with gr.Column():
                srt_type = gr.Radio(["Upload File", "Paste Text"], value="Upload File", label="SRT Input Method")
                file_in = gr.File(label="Upload SRT File", file_types=[".srt"], visible=True)
                text_in = gr.Textbox(label="Paste SRT Text", placeholder="Paste SRT content here...", lines=6, visible=False)
                
                voice_category_srt = gr.Radio(
                    ["Indian Voices 🇮🇳", "Multilingual Voices 🌐", "Other Voices 🌍"], 
                    label="Voice Category", 
                    value="Indian Voices 🇮🇳"
                )
                voice_dropdown_srt = gr.Dropdown(
                    choices=list(INDIAN_VOICES.keys()), 
                    label="Select Voice", 
                    value=list(INDIAN_VOICES.keys())[0] if INDIAN_VOICES else None, 
                    filterable=True
                )
                with gr.Row():
                    rate_srt = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Speed (%)")
                    pitch_srt = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Pitch (Hz)")

                with gr.Group():
                    gr.Markdown("### 🎚️ Studio Export Settings")
                    export_format_srt = gr.Dropdown(["mp3", "wav", "flac", "ogg", "m4a"], label="Format", value="mp3")
                    export_bitrate_srt = gr.Dropdown(["320k", "256k", "192k", "128k", "64k"], label="Bitrate", value="320k")
                    export_hz_srt = gr.Dropdown(["96000Hz", "48000Hz", "44100Hz", "24000Hz"], label="Sample Rate (Hz)", value="48000Hz")
                submit_srt = gr.Button("Convert SRT", variant="primary")
            with gr.Column():
                status_srt = gr.Textbox(label="Status", interactive=False)
                audio_srt = gr.Audio(label="Output Audio", type="filepath")
                file_srt_dl = gr.File(label="Download Audio")
                
                # Dynamic Download Files Row (TXT, CSV, PDF)
                with gr.Row():
                    txt_dl = gr.File(label="Download TXT Log")
                    csv_dl = gr.File(label="Download CSV Log")
                    pdf_dl = gr.File(label="Download PDF Log")
                
                log_srt_grid = gr.Dataframe(label="Google Sheet Log (Target vs Gen Time)")

        srt_type.change(lambda v: (gr.update(visible=v=="Upload File"), gr.update(visible=v=="Paste Text")), inputs=srt_type, outputs=[file_in, text_in])
        voice_category_srt.change(fn=update_voices_by_category, inputs=voice_category_srt, outputs=voice_dropdown_srt)
        
        # Aligned Event Click - Fixed SRT Paste and Visual Progress Bar thread lock
        submit_srt.click(
            fn=process_edge_srt,
            inputs=[
                srt_type, file_in, text_in, 
                voice_dropdown_srt, rate_srt, pitch_srt, 
                export_format_srt, export_bitrate_srt, export_hz_srt
            ],
            outputs=[audio_srt, file_srt_dl, status_srt, log_srt_grid, txt_dl, csv_dl, pdf_dl]
        )

    with gr.Tab("Multi-Speaker SRT"):
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

        # Process multi-speaker compilation (Uses unlimited speaker voice mapping table)
        submit_btn_multi.click(
            fn=process_unlimited_speaker_srt,
            inputs=[
                editor_grid, mapping_grid,
                rate_multi_slider, pitch_multi_slider, multi_export_format, multi_export_bitrate, multi_export_hz
            ],
            outputs=[status_msg_multi, audio_output_multi, file_multi_dl, log_multi_grid, multi_txt_dl, multi_csv_dl, multi_pdf_dl]
        )

    with gr.Tab("Simple Text to Speech"):
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
demo.launch(share=True, debug=True)            export_kwargs["bitrate"] = bitrate
            
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
        
        # Custom elegant green table layout
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

# Parse SRT content safely
def parse_srt_with_timings(srt_content):
    srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
    segments = []
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
            start_ms = time_to_ms(times[0])
            end_ms = time_to_ms(times[1])

            text_lines = lines[timing_line_index + 1:]
            text = " ".join([line.strip() for line in text_lines if line.strip()])

            if text:
                segments.append({'start': start_ms, 'end': end_ms, 'text': text})
        except Exception as e:
            print(f"Warning parsing block: {e}")
            continue
    return segments

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

            # Auto-detect speakers, default to "Speaker 1"
            speaker = "Speaker 1"
            text_content = full_text
            
            match_bracket = re.match(r'^\[\s*(.*?)\s*\]\s*(.*)', full_text)
            match_colon = re.match(r'^(.*?)\s*:\s*(.*)', full_text)
            
            if match_bracket:
                speaker = match_bracket.group(1).strip()
                text_content = match_bracket.group(2).strip()
            elif match_colon:
                # Avoid matching raw timestamps (e.g. HH:MM)
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

# In-memory single chunk generator helper (Zero Disk Writes)
async def fetch_tts_segment_in_memory(sem, text, voice_name, rate_str, pitch_str, idx, log_list=None):
    async with sem:
        start_time = time.time()
        try:
            cleaned_text = clean_text_for_tts(text)
            if not cleaned_text:
                return AudioSegment.silent(duration=100), idx

            communicate = edge_tts.Communicate(cleaned_text, voice_name, rate=rate_str, pitch=pitch_str)
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            audio_buffer.seek(0)
            
            segment_audio = AudioSegment.from_file(audio_buffer, format="mp3")
            gen_time = time.time() - start_time
            if log_list is not None:
                log_list[idx] = gen_time
            return segment_audio, idx
        except Exception as e:
            print(f"Fetch failed for segment: {e}")
            return None, idx

# 1. Single Speaker SRT to Audio Pipeline (with Grid Log & Safe clean)
async def process_edge_srt(srt_type, srt_file, srt_text, voice_selection, speed_val, pitch_val, audio_format, bitrate, sample_rate, progress=gr.Progress()):
    if voice_selection not in ALL_VOICES_MAP:
        yield None, None, "Please select a valid voice.", None, None, None, None
        return

    srt_content = ""
    if srt_type == "Upload File" and srt_file:
        try:
            with open(srt_file.name, 'r', encoding='utf-8') as f:
                srt_content = f.read()
        except Exception as e:
            yield None, None, f"Error reading file: {e}", None, None, None, None
            return
    elif srt_type == "Paste Text" and srt_text:
        srt_content = srt_text

    if not srt_content.strip():
        yield None, None, "No SRT content provided.", None, None, None, None
        return

    voice = ALL_VOICES_MAP[voice_selection]
    start_process_time = time.time()
    segments = parse_srt_with_timings(srt_content)
    combined_audio = AudioSegment.silent(duration=0)
    actual_spoken_ms = 0
    total_segments = len(segments)

    if total_segments == 0:
        yield None, None, "No valid segments found in SRT.", None, None, None, None
        return

    # AI-like Instant Duration Estimation Engine
    total_chars = sum(len(d['text']) for d in segments)
    estimated_compile_time_s = round((total_segments * 0.30) + (total_chars * 0.001), 1)
    
    est_msg = f"🔄 AI Estimate: Total Segments: {total_segments} | Estimated Process Time: ~{estimated_compile_time_s}s. Starting..."
    yield None, None, est_msg, None, None, None, None
    await asyncio.sleep(0.8)

    gen_times = [0.0] * total_segments
    rate_str = f"{speed_val:+d}%"
    pitch_str = f"{pitch_val:+d}%" if pitch_val == 0 else f"{pitch_val:+d}Hz"

    # --- PASS 1: Parallel Estimations ---
    progress(0.0, desc="Pass 1: Estimating segment times...")
    est_audios = [None] * total_segments
    est_tasks = [fetch_tts_segment_in_memory(asyncio.Semaphore(15), s['text'], voice, rate_str, pitch_str, i) for i, s in enumerate(segments)]
    
    completed_est = 0
    for future in asyncio.as_completed(est_tasks):
        audio_segment, idx = await future
        est_audios[idx] = audio_segment
        completed_est += 1
        progress(completed_est / total_segments, desc=f"Pass 1: Estimating ({completed_est}/{total_segments})")

    est_track_duration_s = sum([len(a)/1000 for a in est_audios if a]) / (1 + (speed_val/100))

    # --- PASS 2: Squeeze-to-Fit Generation ---
    progress(0.35, desc=f"Track Est: {est_track_duration_s:.1f}s | Pass 2: Generating...")
    final_audios = [None] * total_segments
    final_tasks = []
    for i, seg in enumerate(segments):
        target = max(1, seg['end'] - seg['start'])
        if est_audios[i] is None: continue
        
        multiplier = len(est_audios[i]) / target
        new_rate = f"{max(-35, min(75, ((1 + (speed_val/100)) * multiplier - 1) * 100)):+.0f}%"
        final_tasks.append(fetch_tts_segment_in_memory(asyncio.Semaphore(15), seg['text'], voice, new_rate, pitch_str, i, gen_times))

    completed_final = 0
    for future in asyncio.as_completed(final_tasks):
        audio_segment, idx = await future
        final_audios[idx] = audio_segment
        completed_final += 1
        progress(completed_final / total_segments, desc=f"Track Est: {est_track_duration_s:.1f}s | Pass 2: Generating ({completed_final}/{total_segments})")
    
    pass2_duration = time.time() - pass2_start if 'pass2_start' in locals() else 0.1

    # --- PASS 3: Absolute Canvas Stitching (0ms Cumulative Drift) ---
    progress(0.8, desc="Pass 3: Stitching absolute timeline...")
    total_canvas_duration = segments[-1]['end'] + 10000
    combined_audio = AudioSegment.silent(duration=total_canvas_duration)

    for i, s in enumerate(segments):
        start_ms = s['start']
        end_ms = s['end']
        target_slot_duration = max(1, end_ms - start_ms)
        speech_audio = final_audios[i]
        if speech_audio is None: continue

        # Strict Overlap Guard: If speech overflows, apply soft fade-out trim
        if len(speech_audio) > target_slot_duration:
            fade_len = min(100, target_slot_duration // 2)
            speech_audio = speech_audio[:target_slot_duration].fade_out(fade_len)

        # Overlay onto exact absolute millisecond position
        combined_audio = combined_audio.overlay(speech_audio, position=start_ms)
        progress(0.8 + (i/total_segments)*0.15, desc="Pass 3: Stitching absolute timeline...")

    # Crop canvas exactly at the real absolute end
    combined_audio = combined_audio[:segments[-1]['end']]

    # Export using High-Quality CD Master
    progress(0.95, desc="Exporting master audio...")
    final_file = export_processed_audio(combined_audio, "final_srt_audio", audio_format, bitrate, sample_rate)
    total_duration = time.time() - start_process_time
    
    # Sateek Wall-Clock Average Segment Generation speed
    avg_segment_time = pass2_duration / total_segments if total_segments else 0

    final_status = (f"✅ Done! Audio Duration: {est_track_duration_s:.2f}s | "
                    f"Total Compile Time: {total_duration:.2f}s | "
                    f"Average Segment Speed: {avg_segment_time:.3f}s")

    # Generate Google Sheets-like Grid Data log
    log_df = pd.DataFrame({
        "Segment Index": range(1, total_segments + 1),
        "Target Duration (s)": [round((s['end']-s['start'])/1000.0, 2) for s in segments],
        "Generation Time (s)": gen_times
    })

    txt_f, csv_f, pdf_f = save_log_formats(log_df)
    yield final_file, final_file, final_status, log_df, txt_f, csv_f, pdf_f

# 2. Unlimited Multi-Speaker SRT to Audio Pipeline
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

    rate_str = f"{speed_val:+d}%"
    pitch_str = f"{pitch_val:+d}%" if pitch_val == 0 else f"{pitch_val:+d}Hz"
    gen_times = [0.0] * total_segments

    # --- PASS 1: Parallel Estimations ---
    progress(0.0, desc="Pass 1: Estimating segment times...")
    est_audios = [None] * total_segments
    est_tasks = []
    for i, seg in enumerate(segments):
        v_name = voice_map.get(seg['speaker'], "hi-IN-SwaraNeural")
        est_tasks.append(fetch_tts_segment_in_memory(asyncio.Semaphore(15), seg['text'], v_name, f"{speed_val:+.0f}%", pitch_str, i))
        
    completed_est = 0
    for future in asyncio.as_completed(est_tasks):
        audio_segment, idx = await future
        est_audios[idx] = audio_segment
        completed_est += 1
        progress(completed_est / total_segments, desc=f"Pass 1: Estimating ({completed_est}/{total_segments})")

    est_track_duration_s = sum([len(a)/1000 for a in est_audios if a]) / (1 + (speed_val/100))

    # --- PASS 2: Squeeze-to-Fit Generation ---
    progress(0.35, desc=f"Track Est: {est_track_duration_s:.1f}s | Pass 2: Generating...")
    final_audios = [None] * total_segments
    final_tasks = []
    for i, s in enumerate(segments):
        target = max(1, s['end'] - s['start'])
        if est_audios[i] is None: continue
        mult = len(est_audios[i]) / target
        nr = f"{max(-35, min(75, ((1 + (speed_val/100)) * mult - 1) * 100)):+.0f}%"
        v_name = voice_map.get(s['speaker'], "hi-IN-SwaraNeural")
        final_tasks.append(fetch_tts_segment_in_memory(asyncio.Semaphore(15), s['text'], v_name, nr, pitch_str, i, gen_times))
        
    pass2_start = time.time()
    completed_final = 0
    for future in asyncio.as_completed(final_tasks):
        audio_segment, idx = await future
        final_audios[idx] = audio_segment
        completed_final += 1
        progress(completed_final / total_segments, desc=f"Track Est: {est_track_duration_s:.1f}s | Pass 2: Generating ({completed_final}/{total_segments})")
        
    pass2_duration = time.time() - pass2_start

    # --- PASS 3: Absolute Canvas Stitching (0ms Cumulative Drift) ---
    progress(0.8, desc="Pass 3: Stitching absolute timeline...")
    total_canvas_duration = segments[-1]['end'] + 10000
    combined_audio = AudioSegment.silent(duration=total_canvas_duration)

    for i, s in enumerate(segments):
        start_ms = s['start']
        end_ms = s['end']
        target_slot_duration = max(1, end_ms - start_ms)
        speech_audio = final_audios[i]
        if speech_audio is None: continue

        # Strict Overlap Guard: If speech overflows, apply soft fade-out trim
        if len(speech_audio) > target_slot_duration:
            fade_len = min(100, target_slot_duration // 2)
            speech_audio = speech_audio[:target_slot_duration].fade_out(fade_len)

        # Overlay onto exact absolute millisecond position
        combined_audio = combined_audio.overlay(speech_audio, position=start_ms)
        progress(0.8 + (i/total_segments)*0.15, desc="Pass 3: Stitching absolute timeline...")

    # Crop canvas exactly at the real absolute end
    combined_audio = combined_audio[:segments[-1]['end']]

    # Export using High-Quality CD Master
    progress(0.95, desc="Exporting master audio...")
    final_file = export_processed_audio(combined_audio, "final_multi_srt_audio", audio_format, bitrate, sample_rate)
    total_duration = time.time() - start_process_time
    
    # Sateek Wall-Clock Average Segment Generation speed
    avg_segment_time = pass2_duration / total_segments if total_segments else 0

    final_status = (f"✅ Done! Audio Duration: {est_track_duration_s:.2f}s | "
                    f"Total Compile Time: {total_duration:.2f}s | "
                    f"Average Segment Speed: {avg_segment_time:.3f}s")

    log_df = pd.DataFrame({
        "Segment Index": range(1, total_segments + 1),
        "Speaker": [s['speaker'] for s in segments],
        "Target Duration (s)": [round((s['end']-s['start'])/1000.0, 2) for s in segments],
        "Generation Time (s)": gen_times
    })

    # Generate log downloads (TXT, CSV, PDF)
    txt_f, csv_f, pdf_f = save_log_formats(log_df)

    yield final_status, final_file, final_file, log_df, txt_f, csv_f, pdf_f

# 3. Simple Text to Speech Pipeline
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
    gr.Markdown("# 🎙️ Professional Studio Quality Voiceover Editor")
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

    with gr.Tab("SRT to Audio (Single Speaker)"):
        with gr.Row():
            with gr.Column():
                srt_type = gr.Radio(["Upload File", "Paste Text"], value="Upload File", label="SRT Input Method")
                file_in = gr.File(label="Upload SRT File", file_types=[".srt"], visible=True)
                text_in = gr.Textbox(label="Paste SRT Text", placeholder="Paste SRT content here...", lines=6, visible=False)
                
                voice_category_srt = gr.Radio(
                    ["Indian Voices 🇮🇳", "Multilingual Voices 🌐", "Other Voices 🌍"], 
                    label="Voice Category", 
                    value="Indian Voices 🇮🇳"
                )
                voice_dropdown_srt = gr.Dropdown(
                    choices=list(INDIAN_VOICES.keys()), 
                    label="Select Voice", 
                    value=list(INDIAN_VOICES.keys())[0] if INDIAN_VOICES else None, 
                    filterable=True
                )
                with gr.Row():
                    rate_srt = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Speed (%)")
                    pitch_srt = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Pitch (Hz)")

                with gr.Group():
                    gr.Markdown("### 🎚️ Studio Export Settings")
                    export_format_srt = gr.Dropdown(["mp3", "wav", "flac", "ogg", "m4a"], label="Format", value="mp3")
                    export_bitrate_srt = gr.Dropdown(["320k", "256k", "192k", "128k", "64k"], label="Bitrate", value="320k")
                    export_hz_srt = gr.Dropdown(["96000Hz", "48000Hz", "44100Hz", "24000Hz"], label="Sample Rate (Hz)", value="48000Hz")
                submit_srt = gr.Button("Convert SRT", variant="primary")
            with gr.Column():
                status_srt = gr.Textbox(label="Status", interactive=False)
                audio_srt = gr.Audio(label="Output Audio", type="filepath")
                file_srt_dl = gr.File(label="Download Audio")
                
                # Dynamic Download Files Row (TXT, CSV, PDF)
                with gr.Row():
                    txt_dl = gr.File(label="Download TXT Log")
                    csv_dl = gr.File(label="Download CSV Log")
                    pdf_dl = gr.File(label="Download PDF Log")
                
                log_srt_grid = gr.Dataframe(label="Google Sheet Log (Target vs Gen Time)")

        srt_type.change(lambda v: (gr.update(visible=v=="Upload File"), gr.update(visible=v=="Paste Text")), inputs=srt_type, outputs=[file_in, text_in])
        voice_category_srt.change(fn=update_voices_by_category, inputs=voice_category_srt, outputs=voice_dropdown_srt)
        
        submit_srt.click(
            fn=process_edge_srt,
            inputs=[
                srt_type, file_in, text_in, 
                voice_dropdown_srt, rate_srt, pitch_srt, 
                export_format_srt, export_bitrate_srt, export_hz_srt
            ],
            outputs=[audio_out := audio_srt, file_srt_dl, status_srt, log_srt_grid, txt_dl, csv_dl, pdf_dl]
        )

    with gr.Tab("Multi-Speaker SRT"):
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

        # Load data and dynamically generate speaker mapping rows on load click
        load_m.click(
            fn=load_srt_and_auto_extract_speakers,
            inputs=[m_in_type, m_file, m_text],
            outputs=[editor_grid, mapping_grid]
        )

        # Process multi-speaker compilation (Uses unlimited speaker voice mapping table)
        submit_btn_multi.click(
            fn=process_unlimited_speaker_srt,
            inputs=[
                editor_grid, mapping_grid,
                rate_multi_slider, pitch_multi_slider, multi_export_format, multi_export_bitrate, multi_export_hz
            ],
            outputs=[status_msg_multi, audio_output_multi, file_multi_dl, log_multi_grid, multi_txt_dl, multi_csv_dl, multi_pdf_dl]
        )

    with gr.Tab("Simple Text to Speech"):
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
demo.launch(share=True, debug=True)

!pip install -q edge-tts pydub nest-asyncio
import gradio as gr
import asyncio
import edge_tts
import os
import re
import time
import nest_asyncio
import pandas as pd
from pydub import AudioSegment
from IPython.display import FileLink, display

nest_asyncio.apply()

# --- VIBRANT GREEN UI DESIGN ---
green_theme = gr.themes.Default(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="slate"
)

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

async def process_edge_srt(srt_text, voice, audio_format, bitrate, sample_rate):
    if not srt_text.strip():
        yield None, None, "Please paste SRT text.", None
        return

    start_process_time = time.time()
    segments = [s.strip() for s in re.split(r'\n\s*\n', srt_text.strip()) if s.strip()]
    combined_audio = AudioSegment.silent(duration=0)
    actual_spoken_ms = 0

    parsed_data = []
    for segment in segments:
        lines = [l.strip() for l in segment.split('\n') if l.strip()]
        time_line = next((l for l in lines if "-->" in l), None)
        if time_line:
            time_match = re.search(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,\.]\d{3})', time_line)
            if time_match:
                start_ms = time_to_ms(time_match.group(1))
                end_ms = time_to_ms(time_match.group(2))
                text_idx = lines.index(time_line) + 1
                text = " ".join(lines[text_idx:])
                if text.strip():
                    parsed_data.append({'start': start_ms, 'end': end_ms, 'text': text})

    total_segments = len(parsed_data)
    if total_segments == 0:
        yield None, None, "No valid segments found in SRT.", None
        return

    total_chars = sum(len(d['text']) for d in parsed_data)
    estimated_compile_time_s = round((total_segments * 0.30) + (total_chars * 0.001), 1)

    est_msg = f"🔄 AI Estimate: Total Segments: {total_segments} | Estimated Process Time: ~{estimated_compile_time_s}s. Starting..."
    yield None, None, est_msg, None
    await asyncio.sleep(0.8)

    gen_times = []
    for i, data in enumerate(parsed_data):
        current_msg = f"Processing segment {i+1} of {total_segments}... [Est. Time Remaining: {round(max(0, estimated_compile_time_s - (i * 0.35)), 1)}s]"
        yield None, None, current_msg, None

        start_seg_time = time.time()
        temp_file = f"gradio_temp_{i}.mp3"
        communicate = edge_tts.Communicate(data['text'], voice)
        await communicate.save(temp_file)

        if os.path.exists(temp_file):
            seg_audio = AudioSegment.from_file(temp_file, format="mp3")
            target_start = data['start']
            target_end = data['end']
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

    final_file = export_processed_audio(combined_audio, "final_srt_audio", audio_format, bitrate, sample_rate)

    total_timeline_sec = round(len(combined_audio) / 1000.0, 2)
    spoken_only_sec = round(actual_spoken_ms / 1000.0, 2)
    end_process_time = time.time()
    elapsed_generation = round(end_process_time - start_process_time, 2)
    avg_time = round(elapsed_generation / total_segments, 2) if total_segments > 0 else 0

    final_status = (f"✅ Done! Spoken Audio: {spoken_only_sec}s | Total Timeline: {total_timeline_sec}s\n"
                    f"Generated in {elapsed_generation}s (Avg: {avg_time}s/seg).")

    log_df = pd.DataFrame({
        "Segment Index": range(1, total_segments + 1),
        "Target Duration (s)": [round((s['end']-s['start'])/1000.0, 2) for s in parsed_data],
        "Generation Time (s)": gen_times
    })

    yield final_file, final_file, final_status, log_df

async def get_categorized_voices():
    all_v = await edge_tts.list_voices()
    multi, indian, other = [], [], []
    indian_locales = ['hi-IN', 'en-IN', 'bn-IN', 'gu-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'ta-IN', 'te-IN', 'ur-IN']
    for v in all_v:
        label, value = f"{v['FriendlyName']} ({v['Locale']})", v['ShortName']
        if "Multilingual" in v['FriendlyName']: multi.append((label, value))
        elif any(loc in v['Locale'] for loc in indian_locales): indian.append((label, value))
        else: other.append((label, value))
    return sorted(multi), sorted(indian), sorted(other), len(all_v)

async def start_app():
    multi_v, indian_v, other_v, total_voices = await get_categorized_voices()
    with gr.Blocks(title="Edge TTS Full Voice Access", theme=green_theme) as demo:
        gr.Markdown(f"# 🎙️ Edge TTS - All {total_voices} Global Voices")
        with gr.Row():
            with gr.Column():
                srt_input = gr.Textbox(label="Paste SRT Content", lines=8)
                with gr.Column() as manual_box:
                    with gr.Tabs():
                        with gr.TabItem("🇮🇳 Indian"):
                            v_ind = gr.Dropdown(label=f"Indian Voices ({len(indian_v)})", choices=indian_v, value="hi-IN-SwaraNeural")
                        with gr.TabItem("🌐 Multilingual"):
                            v_mul = gr.Dropdown(label=f"Multilingual Voices ({len(multi_v)})", choices=multi_v)
                        with gr.TabItem("🌎 Global"):
                            v_oth = gr.Dropdown(label=f"Global Voices ({len(other_v)})", choices=other_v)

                current_voice = gr.State("hi-IN-SwaraNeural")
                v_ind.change(lambda v: v, v_ind, current_voice)
                v_mul.change(lambda v: v, v_mul, current_voice)
                v_oth.change(lambda v: v, v_oth, current_voice)

                # ADD-ON: Studio Export Settings
                with gr.Group():
                    gr.Markdown("### 🎚️ Studio Export Settings")
                    export_format_srt = gr.Dropdown(["mp3", "wav", "flac", "ogg", "m4a"], label="Format", value="mp3")
                    export_bitrate_srt = gr.Dropdown(["320k", "256k", "192k", "128k", "64k"], label="Bitrate", value="320k")
                    export_hz_srt = gr.Dropdown(["96000Hz", "48000Hz", "44100Hz", "24000Hz"], label="Sample Rate (Hz)", value="48000Hz")

                submit_btn = gr.Button("Generate Audio", variant="primary")

            with gr.Column():
                audio_out = gr.Audio(label="Preview Audio", type="filepath")
                file_out = gr.File(label="Download Audio")
                status_out = gr.Textbox(label="Progress")
                log_srt_grid = gr.Dataframe(label="Google Sheet Log (Target vs Gen Time)")

        submit_btn.click(
            fn=process_edge_srt,
            inputs=[
                srt_input, current_voice,
                export_format_srt, export_bitrate_srt, export_hz_srt
            ],
            outputs=[audio_out, file_out, status_out, log_srt_grid]
        )

    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    await start_app()

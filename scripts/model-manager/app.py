"""
app.py — Gradio UI for Discrete Diffusion Speech Translation.

Loads the discrete diffusion model and provides a web interface to translate
Vietnamese speech to English, Chinese, and Korean, with a detailed, step-by-step
visualization of the forward (add noise) and reverse (denoise) diffusion processes.
"""

import sys
import os
import torch
import numpy as np
import gradio as gr

# ─── Resolve project root so we can import src/ modules ──────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from model.configuration_dlm import DiscreteDiffusionConfig
from model.modeling_dlm import DiscreteDiffusionModel, decoder_out_t
from huggingface_hub import hf_hub_download
import json

# ─── Global variables for preloaded model/tokenizer ─────────────────────────
_tokenizer = None
_model = None
_feature_extractor = None
_device = None

def get_model_and_tokenizer():
    global _tokenizer, _model, _feature_extractor, _device
    if _model is None:
        repo_id = "aiai-laboratory/diffusion-speech-translation-from-vi-v1"
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[app] Using device: {_device}")
        
        print(f"[app] Loading tokenizer from {repo_id}...")
        _tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, use_fast=False)
        
        print(f"[app] Loading config from {repo_id}...")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        config = DiscreteDiffusionConfig(**{
            k: v for k, v in config_dict.items()
            if not k.startswith("_") and k != "model_type" and k != "transformers_version" and k != "auto_map"
        })
        
        print("[app] Building DiscreteDiffusionModel from config...")
        _model = DiscreteDiffusionModel(config)
        
        print("[app] Downloading and loading weights...")
        try:
            weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device="cpu")
        except Exception:
            weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            
        _model.load_state_dict(state_dict, strict=False)
        
        if config.tie_word_embeddings:
            _model.model.lm_head.decoder.weight = _model.model.roberta.embeddings.word_embeddings.weight
            
        _model = _model.eval().to(_device)
        if _device == "cuda":
            _model = _model.to(torch.bfloat16)
            print("[app] Model cast to bfloat16")
            
        print("[app] Loading feature extractor...")
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.audio_encoder_name)
        print("[app] Preloading complete.")
        
    return _tokenizer, _model, _feature_extractor, _device

# ─── Audio loader ────────────────────────────────────────────────────────────
def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and return resampled float32 numpy array."""
    waveform = None
    sr = None

    try:
        import soundfile as sf
        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
    except Exception:
        pass

    if waveform is None:
        try:
            import librosa
            waveform, sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        except Exception:
            pass

    if waveform is None:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(target_sr)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            waveform = samples / (2 ** (audio.sample_width * 8 - 1))
            sr = target_sr
        except Exception:
            pass

    if waveform is None:
        raise RuntimeError(f"Could not load audio file '{path}'. Please upload a valid WAV, MP3, or FLAC file.")

    if sr != target_sr:
        ratio = target_sr / sr
        new_length = int(len(waveform) * ratio)
        indices = np.linspace(0, len(waveform) - 1, new_length)
        waveform = np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)

    return waveform

# ─── HTML formatting helpers ──────────────────────────────────────────────────
def tokens_to_html_denoise_single_line(canvas_steps, tokenizer, step_idx, max_iterations):
    mask_id = tokenizer.mask_token_id
    curr_tokens = canvas_steps[-1]
    prev_tokens = canvas_steps[-2] if len(canvas_steps) > 1 else None
    
    # Progress percent
    progress_percent = int((step_idx / max_iterations) * 100) if max_iterations > 0 else 100
    
    # Count masks and tokens
    num_masks = sum(1 for c in curr_tokens if c == mask_id)
    num_tokens = len(curr_tokens)
    denoised_count = num_tokens - num_masks
    
    header_html = f"""
    <div class="console-header">
        <div class="console-status-group">
            <span class="status-dot {'status-pulse' if step_idx < max_iterations else 'status-done'}"></span>
            <span class="console-title">Reverse Process (Denoising) — Step {step_idx}/{max_iterations}</span>
        </div>
        <div class="console-progress-container">
            <div class="console-progress-bar" style="width: {progress_percent}%;"></div>
        </div>
        <span class="console-stats">{denoised_count}/{num_tokens} tokens decoded</span>
    </div>
    """
    
    tokens_html = []
    for i, c_id in enumerate(curr_tokens):
        if c_id == mask_id:
            tokens_html.append('<span class="token token-mask">░░░</span>')
        else:
            token_str = tokenizer.decode([c_id]).strip()
            if not token_str:
                token_str = " "
            # Check if this token was a mask in the previous step
            if prev_tokens is not None and i < len(prev_tokens) and prev_tokens[i] == mask_id:
                tokens_html.append(f'<span class="token token-new">{token_str}</span>')
            else:
                tokens_html.append(f'<span class="token token-normal">{token_str}</span>')
                
    joined_tokens = "".join(tokens_html)
    
    html = f"""
    <div class="realtime-console">
        {header_html}
        <div class="console-line">
            {joined_tokens}
        </div>
    </div>
    """
    return html


# ─── Core translation function ───────────────────────────────────────────────
def run_speech_translation(audio_path, max_iterations, strategy, slow_mode_enabled):
    import time
    if not audio_path:
        yield "Please select or record an audio file.", "", "", "", "", ""
        return
        
    try:
        tokenizer, model, feature_extractor, device = get_model_and_tokenizer()
        
        # Load and preprocess audio
        print(f"[app] Loading audio from: {audio_path}")
        waveform = load_audio(audio_path, target_sr=16000)
        audio_duration = len(waveform) / 16000
        
        audio_inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
        audio_values_raw = audio_inputs.input_values.to(device)
        
        audio_len = audio_values_raw.size(-1)
        padded_len = ((audio_len + 79) // 80) * 80
        padded_audio = torch.zeros(1, padded_len, device=device)
        padded_audio[0, :audio_len] = audio_values_raw[0]
        audio_values = padded_audio
        
        padded_mask = torch.zeros(1, padded_len, dtype=torch.long, device=device)
        padded_mask[0, :audio_len] = 1
        audio_attention_mask = padded_mask
        
        if device == "cuda":
            audio_values = audio_values.to(torch.bfloat16)
            
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        mask_id = tokenizer.mask_token_id
        pad_id = tokenizer.pad_token_id
        
        TASKS = {
            "english": "<vi_en>",
            "chinese": "<vi_zh>",
            "korean":  "<vi_ko>",
        }
        
        state = {}
        with torch.no_grad():
            for lang, task_token in TASKS.items():
                task_token_id = tokenizer.convert_tokens_to_ids(task_token)
                
                # Heuristic canvas length based on duration
                if lang == "english":
                    canvas_len = int(audio_duration * 4.0)
                    canvas_len = max(5, min(100, canvas_len))
                else:
                    canvas_len = int(audio_duration * 2.5)
                    canvas_len = max(5, min(64, canvas_len))
                
                # Setup canvas
                src_tokens = torch.tensor([[bos_id, task_token_id]], dtype=torch.long, device=device)
                src_length = src_tokens.size(1)
                
                canvas = torch.cat([
                    src_tokens,
                    torch.full((1, canvas_len), mask_id, dtype=torch.long, device=device),
                    torch.tensor([[eos_id]], dtype=torch.long, device=device),
                ], dim=1)
                
                partial_mask = torch.zeros_like(canvas, dtype=torch.bool)
                partial_mask[:, :src_length] = True
                
                non_fixed_sym_masks = (
                    canvas.ne(pad_id) &
                    canvas.ne(bos_id) &
                    canvas.ne(eos_id) &
                    ~partial_mask
                )
                
                output_scores = torch.zeros_like(canvas, dtype=torch.float32)
                output_mask = canvas.eq(mask_id)
                
                decoder_out = decoder_out_t(
                    output_tokens=canvas.clone(),
                    output_scores=output_scores,
                    output_masks=output_mask,
                    non_fixed_sym_masks=non_fixed_sym_masks,
                    attn=None,
                    step=0,
                    max_step=max_iterations,
                    history=[], # Always collect history to display
                )
                
                state[lang] = {
                    "decoder_out": decoder_out,
                    "partial_mask": partial_mask,
                    "canvas_len": canvas_len,
                    "history": [canvas.clone()], # Step 0 is the initial canvas
                    "final_translation": "",
                    "final_tokens": None
                }

            # Yield Step 0 (all masks) first to visualize initial state
            curr_translations = {"english": "", "chinese": "", "korean": ""}
            denoise_htmls = {}
            for lang in TASKS:
                lang_state = state[lang]
                canvas_steps = [[mask_id] * lang_state["canvas_len"]]
                denoise_htmls[lang] = tokens_to_html_denoise_single_line(
                    canvas_steps,
                    tokenizer,
                    0,
                    max_iterations
                )
            yield (
                curr_translations["english"],
                curr_translations["chinese"],
                curr_translations["korean"],
                denoise_htmls["english"],
                denoise_htmls["chinese"],
                denoise_htmls["korean"]
            )
            time.sleep(0.1)

            # 1. Reverse Denoising process
            for step_idx in range(max_iterations):
                for lang in TASKS:
                    lang_state = state[lang]
                    decoder_out = model.denoise_step(
                        lang_state["decoder_out"],
                        lang_state["partial_mask"],
                        temperature=1.0,
                        strategy=strategy,
                        audio_features=audio_values,
                        audio_attention_mask=audio_attention_mask,
                    )
                    lang_state["decoder_out"] = decoder_out
                    lang_state["history"].append(decoder_out.output_tokens.clone())
                    
                # Decode intermediate tokens and yield after each step
                curr_translations = {}
                denoise_htmls = {}
                for lang in TASKS:
                    lang_state = state[lang]
                    out_tokens = lang_state["decoder_out"].output_tokens[0]
                    cutoff = (
                        out_tokens.ne(pad_id) &
                        out_tokens.ne(bos_id) &
                        out_tokens.ne(eos_id) &
                        ~lang_state["partial_mask"][0]
                    )
                    gen_tokens = out_tokens[cutoff]
                    curr_translations[lang] = tokenizer.decode(gen_tokens.cpu(), skip_special_tokens=True).strip()
                    
                    canvas_steps = [[mask_id] * lang_state["canvas_len"]]
                    for step_tokens in lang_state["history"][1:]:
                        tokens_list = step_tokens[0][~lang_state["partial_mask"][0]].tolist()[:lang_state["canvas_len"]]
                        canvas_steps.append(tokens_list)
                        
                    denoise_htmls[lang] = tokens_to_html_denoise_single_line(
                        canvas_steps,
                        tokenizer,
                        step_idx + 1,
                        max_iterations
                    )
                    
                yield (
                    curr_translations["english"],
                    curr_translations["chinese"],
                    curr_translations["korean"],
                    denoise_htmls["english"],
                    denoise_htmls["chinese"],
                    denoise_htmls["korean"]
                )
                if slow_mode_enabled:
                    time.sleep(0.2)
                
    except Exception as e:
        import traceback
        err_msg = f"An error occurred during inference:\n{str(e)}\n\n{traceback.format_exc()}"
        print(err_msg)
        yield err_msg, "", "", "", "", ""

# ─── Load test sample helper ─────────────────────────────────────────────────
def load_test_sample():
    sample_path = os.path.join(PROJECT_ROOT, "test/test_data/test_sample.mp3")
    if os.path.exists(sample_path):
        return sample_path
    return None

# ─── UI Styling and CSS ───────────────────────────────────────────────────────
CSS = """
/* Real-time Console Styling */
.realtime-console {
    background-color: #0f172a; /* Slate 900 */
    border: 1px solid #334155; /* Slate 700 */
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
    margin-bottom: 16px;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.console-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1e293b;
    flex-wrap: wrap;
    gap: 8px;
}

.console-status-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.status-pulse {
    background-color: #10b981; /* Emerald 500 */
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-green 1.5s infinite;
}

.status-done {
    background-color: #3b82f6; /* Blue 500 */
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.5);
}

@keyframes pulse-green {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
}

.console-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #e2e8f0; /* Slate 200 */
}

.console-progress-container {
    flex-grow: 1;
    max-width: 200px;
    height: 6px;
    background-color: #1e293b;
    border-radius: 9999px;
    overflow: hidden;
    margin: 0 16px;
}

.console-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #3b82f6);
    border-radius: 9999px;
    transition: width 0.2s ease-out;
}

.console-stats {
    font-size: 0.8rem;
    font-weight: 500;
    color: #94a3b8; /* Slate 400 */
}

.console-line {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    min-height: 48px;
    padding: 10px 12px;
    background-color: #020617; /* Slate 950 */
    border-radius: 8px;
    border: 1px solid #1e293b;
}

.token {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 4px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', Courier, monospace;
    font-size: 0.9rem;
    transition: all 0.2s ease;
}

.token-mask {
    color: #475569; /* Slate 600 */
    background-color: #1e293b; /* Slate 800 */
    border: 1px dashed #334155;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.token-new {
    color: #059669; /* Emerald 600 */
    background-color: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    font-weight: 600;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.15);
    transform: scale(1.05);
}

.token-normal {
    color: #f1f5f9; /* Slate 100 */
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
"""

# ─── Gradio layout building ───────────────────────────────────────────────────
def build_interface():
    # Pre-trigger loading of the model so weights are ready when server starts
    print("[app] Preloading model on startup...")
    get_model_and_tokenizer()
    
    with gr.Blocks(title="Discrete Diffusion Speech Translation") as demo:
        gr.Markdown("# 🎙️ Vietnamese Speech Translation & Discrete Diffusion Visualizer")
        gr.Markdown(
            "Translate Vietnamese speech directly into three target languages: **English, Chinese, and Korean** "
            "using the Discrete Diffusion model. Record or upload an audio file to start."
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Input Audio (Vietnamese Speech)"
                )
                
                load_sample_btn = gr.Button("📂 Use sample file (test_sample.mp3)", variant="secondary")
                
                with gr.Accordion("Advanced Options", open=False):
                    iterations = gr.Slider(
                        minimum=5, maximum=50, value=10, step=1,
                        label="Denoising steps"
                    )
                    strategy = gr.Dropdown(
                        choices=[
                            "reparam-uncond-deterministic-cosine",
                            "reparam-uncond-stochastic0.1-cosine",
                            "cmlm",
                            "ar"
                        ],
                        value="reparam-uncond-deterministic-cosine",
                        label="Decoding Strategy"
                    )
                
                slow_mode = gr.Checkbox(label="Slow mode (step-by-step visualization)", value=True)
                translate_btn = gr.Button("🚀 Translate", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Translation Results")
                with gr.Row():
                    en_output = gr.Textbox(label="🇬🇧 English", interactive=False)
                    zh_output = gr.Textbox(label="🇨🇳 Chinese", interactive=False)
                    ko_output = gr.Textbox(label="🇰🇷 Korean", interactive=False)
                    
        # Toggleable visualization block
        with gr.Row(visible=True) as viz_section:
            with gr.Column():
                gr.Markdown("### 🔍 Step-by-Step Diffusion Process Visualization")
                with gr.Tabs():
                    with gr.Tab("🇬🇧 English"):
                        gr.Markdown("#### Reverse Process (Denoising)")
                        en_denoise_html = gr.HTML()
                                
                    with gr.Tab("🇨🇳 Chinese"):
                        gr.Markdown("#### Reverse Process (Denoising)")
                        zh_denoise_html = gr.HTML()
                                
                    with gr.Tab("🇰🇷 Korean"):
                        gr.Markdown("#### Reverse Process (Denoising)")
                        ko_denoise_html = gr.HTML()
                                
        # Click listeners
        load_sample_btn.click(fn=load_test_sample, outputs=audio_input)
        
        translate_btn.click(
            fn=run_speech_translation,
            inputs=[audio_input, iterations, strategy, slow_mode],
            outputs=[
                en_output, zh_output, ko_output,
                en_denoise_html,
                zh_denoise_html,
                ko_denoise_html
            ]
        )
        
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        theme=gr.themes.Soft(),
        css=CSS
    )

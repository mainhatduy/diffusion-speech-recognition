import os
import dotenv
import qai_hub as hub

def main():
    dotenv.load_dotenv()
    token = os.getenv("QUALCOMM_TOKEN")
    if not token:
        print("QUALCOMM_TOKEN not found in .env")
        return
        
    os.environ["QAI_HUB_API_TOKEN"] = token
    
    print("Initializing qai_hub client...")
    client = hub.Client()
    
    device = hub.Device("Samsung Galaxy S25 (Family)")
    print(f"Target device: {device.name}")
    
    print("Testing submission for audio_encoder...")
    try:
        # We submit a compile job for audio_encoder first
        job = hub.submit_compile_job(
            model="onnx/audio_encoder_pkg.onnx",
            device=device,
            input_specs={
                "audio_features": (1, 2400),
                "audio_attention_mask": ((1, 2400), "int64")
            },
            options="--target_runtime qnn_context_binary",
            name="audio_encoder_qnn_test"
        )
        print(f"Audio encoder compile job submitted: {job.url}")
        print("Status:", job.get_status())
    except Exception as e:
        print(f"Failed to submit audio_encoder: {e}")
        
    print("\nTesting submission for diffusion_backbone...")
    try:
        job_backbone = hub.submit_compile_job(
            model="onnx/diffusion_backbone_pkg.onnx",
            device=device,
            input_specs={
                "prev_output_tokens": ((1, 32), "int64"),
                "partial_mask": (1, 32), # testing shape-only for bool input
                "precomputed_audio_embeds": (1, 96, 768),
                "precomputed_audio_mask": ((1, 96), "int32")
            },
            options="--target_runtime qnn_context_binary",
            name="diffusion_backbone_qnn_test"
        )
        print(f"Diffusion backbone compile job submitted: {job_backbone.url}")
        print("Status:", job_backbone.get_status())
    except Exception as e:
        print(f"Failed to submit diffusion_backbone: {e}")

if __name__ == "__main__":
    main()

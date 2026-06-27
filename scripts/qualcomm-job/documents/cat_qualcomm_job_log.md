python -c "
import os
import dotenv
dotenv.load_dotenv()
token = os.getenv('QUALCOMM_TOKEN')
os.environ['QAI_HUB_API_TOKEN'] = token
import qai_hub as hub

os.makedirs('onnx/logs', exist_ok=True)
job = hub.get_job('<job_id>')
job.download_job_logs('onnx/logs/audio_encoder_compile_failed')
print('Logs downloaded to onnx/logs/audio_encoder_compile_failed')
" 2>&1 && ls -la onnx/logs/audio_encoder_compile_failed/ && cat onnx/logs/audio_encoder_compile_failed/*.txt 2>/dev/null || cat onnx/logs/audio_encoder_compile_failed/*.log 2>/dev/null || ls onnx/logs/audio_encoder_compile_failed/

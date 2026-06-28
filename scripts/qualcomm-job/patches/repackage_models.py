import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import repackage_model

def main():
    # Repackage Audio Encoder
    repackage_model(
        model_path="onnx/audio_encoder.onnx",
        output_dir="onnx/audio_encoder_pkg.onnx",
        model_name="audio_encoder.onnx",
        data_name="audio_encoder.data"
    )
    
    # Repackage Diffusion Backbone
    repackage_model(
        model_path="onnx/diffusion_backbone.onnx",
        output_dir="onnx/diffusion_backbone_pkg.onnx",
        model_name="diffusion_backbone.onnx",
        data_name="diffusion_backbone.data"
    )
    print("Repackaging complete!")

if __name__ == "__main__":
    main()


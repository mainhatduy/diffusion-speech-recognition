import os
import onnx

def repackage_model(model_path, output_dir, model_name, data_name):
    print(f"Repackaging {model_path} to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = onnx.load(model_path)
    
    # Save the model with external data
    target_model_path = os.path.join(output_dir, model_name)
    onnx.save(
        model,
        target_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name
    )
    print(f"Successfully saved to {target_model_path} and {os.path.join(output_dir, data_name)}")

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

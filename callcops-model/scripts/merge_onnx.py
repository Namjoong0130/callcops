
import onnx
import sys
from pathlib import Path

def merge_onnx(model_path, output_path):
    print(f"Loading {model_path}...")
    # load_external_data=True is default
    model = onnx.load(str(model_path))
    
    print(f"Saving to {output_path} (self-contained)...")
    # serialization will embed data if size allows (<2GB) and we don't ask for external
    onnx.save_model(model, str(output_path))
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_onnx.py <input> <output>")
        sys.exit(1)
        
    merge_onnx(sys.argv[1], sys.argv[2])

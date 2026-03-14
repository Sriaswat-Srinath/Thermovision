"""Quick test to diagnose the /detect/image 500 error."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Create a tiny valid JPEG in memory
import numpy as np
import cv2

# Make a small 100x100 test image
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[30:70, 30:70] = [0, 128, 255]  # orange square
_, buf = cv2.imencode('.jpg', img)
jpg_bytes = buf.tobytes()

print(f"Test image: {len(jpg_bytes)} bytes")

# Try importing and running the detection logic directly
try:
    from app.main import get_models, run_fusion, rgb_to_ir, decode
    print("Imports OK")
    
    # Decode test image
    rgb = decode(jpg_bytes)
    print(f"Decoded: shape={rgb.shape}")
    
    # Generate IR
    ir = rgb_to_ir(rgb)
    print(f"IR generated: shape={ir.shape}")
    
    # Load models
    print("Loading models...")
    rgb_model, ir_model = get_models()
    print("Models loaded OK")
    
    # Run fusion
    print("Running fusion...")
    result = run_fusion(rgb, ir)
    print(f"Fusion OK: rgb={result['rgb_count']}, ir={result['ir_count']}, fusion={result['fusion_count']}")
    
except Exception as e:
    import traceback
    print(f"\n*** ERROR ***")
    traceback.print_exc()

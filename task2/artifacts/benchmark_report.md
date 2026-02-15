# Benchmark Report: PyTorch vs ONNX
- Audio: `samples\sample.wav`
- Runs: `20` (average latency)

## Model size (MB)
- PyTorch checkpoint folder: **0.0 MB**
- ONNX (float): **360.32 MB**
- ONNX (int8): **116.28 MB**

## Inference time (seconds, average)
- PyTorch: **1.5180 s**
- ONNX (float): **1.5463 s**
- ONNX (int8): **0.9125 s**

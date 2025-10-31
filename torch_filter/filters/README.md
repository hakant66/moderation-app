# TorchScript Artifacts

Place your TorchScript models for the torch_filter microservice here:

- `model.ts`        — image classifier TorchScript (MobileNet-sized input)
- `text_model.ts`   — text classifier TorchScript (forward(input_ids, attention_mask))

For local wiring/tests, you can generate stubs:
```bash
python ../export_stub_image.py
python ../export_stub_text.py

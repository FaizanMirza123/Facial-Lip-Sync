# Quick Start Guide

This guide will help you get started with the Facial Lip-Sync system.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- FFmpeg installed on your system

## Installation

1. Create and activate virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download face detection models (first run will download automatically)

## Usage

### 1. Prepare Training Data

Organize your video files and prepare them for training:

```bash
python examples/prepare_data.py --input raw_videos/ --output data/train/
```

### 2. Train the Model

Train your custom lip-sync model:

```bash
python examples/train_model.py --data_dir data/train/ --device cuda
```

Training will create checkpoints in `checkpoints/` directory.

### 3. Run Inference

#### Simple Inference (Image + Audio → Video)

```bash
python examples/run_inference.py ^
    --image face.jpg ^
    --audio speech.wav ^
    --output result.mp4
```

#### With Text-to-Speech

```bash
python examples/run_inference.py ^
    --image face.jpg ^
    --text "Hello, this is a test!" ^
    --output result.mp4 ^
    --use-tts
```

#### Full Conversational AI Demo

```bash
python examples/demo_conversation.py ^
    --image face.jpg ^
    --input-audio user_question.wav ^
    --output ai_response.mp4
```

## Configuration

Edit `configs/config.yaml` to customize:

- Model architecture
- Training hyperparameters
- Audio processing settings
- STT/TTS providers

## Data Requirements

For training:

- Video clips with clear facial views
- 10-50 hours of diverse speakers recommended
- Videos should be 2-10 seconds long
- Good audio quality
- Various lighting conditions

## Tips for Best Results

1. **Training Data**: Use diverse speakers and expressions
2. **Image Quality**: Use high-resolution, well-lit face images
3. **Audio Quality**: Clear speech with minimal background noise
4. **Model Selection**: Start with base Whisper model, upgrade if needed
5. **GPU Memory**: Reduce batch size if you encounter OOM errors

## Troubleshooting

### Face Not Detected

- Ensure face is clearly visible and frontal
- Try different face detector: edit `config.yaml` detector setting

### Poor Lip-Sync

- Train longer or with more data
- Check audio quality
- Adjust loss weights in config

### Out of Memory

- Reduce batch size in config
- Use smaller model backbone
- Process on CPU (slower)

## Next Steps

- Fine-tune on your specific speaker for best results
- Integrate with your LLM of choice
- Experiment with different voices in TTS
- Try real-time streaming (advanced)

## Support

For issues and questions, check the README.md for more details.

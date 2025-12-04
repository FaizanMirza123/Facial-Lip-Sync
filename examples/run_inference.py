"""
Example script for running inference with the lip-sync model.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.realtime_sync import RealtimeLipSync
from stt_tts.whisper_stt import STTManager
from stt_tts.tts_integration import TTSManager


def run_simple_inference(args, config):
    """Run simple inference with face image and audio."""
    print("\n" + "="*50)
    print("Running Simple Inference")
    print("="*50)
    
    # Create lip sync engine
    print("Loading model...")
    lip_sync = RealtimeLipSync(
        model_path=args.model,
        config=config,
        device=args.device
    )
    
    # Set face image
    print(f"Loading face image: {args.image}")
    lip_sync.set_face_image(args.image)
    
    # Process audio
    print(f"Processing audio: {args.audio}")
    lip_sync.process_audio_file(args.audio, args.output)
    
    print("\n" + "="*50)
    print(f"Success! Output saved to: {args.output}")
    print("="*50)


def run_with_stt_tts(args, config):
    """Run inference with STT/TTS integration."""
    print("\n" + "="*50)
    print("Running with STT/TTS Integration")
    print("="*50)
    
    # Initialize STT
    print("Initializing STT...")
    stt = STTManager(
        provider=config['stt_tts']['stt']['provider'],
        config={
            'model_name': config['stt_tts']['stt']['model'],
            'language': config['stt_tts']['stt']['language']
        }
    )
    
    # Transcribe input audio
    if args.input_audio:
        print(f"\nTranscribing input: {args.input_audio}")
        transcription = stt.transcribe(args.input_audio)
        print(f"Transcription: {transcription}")
    else:
        transcription = args.text
    
    # Get LLM response (placeholder - integrate with your LLM)
    print("\nGenerating response...")
    response_text = args.response or f"Response to: {transcription}"
    print(f"Response text: {response_text}")
    
    # Initialize TTS
    print("\nInitializing TTS...")
    tts = TTSManager(
        provider=config['stt_tts']['tts']['provider'],
        config={
            'model_name': config['stt_tts']['tts']['model']
        }
    )
    
    # Generate speech
    temp_audio = "temp_response.wav"
    print("Synthesizing speech...")
    tts.synthesize(response_text, temp_audio)
    
    # Create lip-synced video
    print("\nGenerating lip-synced video...")
    lip_sync = RealtimeLipSync(
        model_path=args.model,
        config=config,
        device=args.device
    )
    
    lip_sync.set_face_image(args.image)
    lip_sync.process_audio_file(temp_audio, args.output)
    
    # Cleanup
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    print("\n" + "="*50)
    print(f"Success! Output saved to: {args.output}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Run lip-sync inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple inference with image and audio
  python run_inference.py --image face.jpg --audio speech.wav --output result.mp4
  
  # With STT/TTS
  python run_inference.py --image face.jpg --text "Hello world" --output result.mp4 --use-tts
  
  # Process existing audio with transcription
  python run_inference.py --image face.jpg --input-audio input.wav --response "Your response" --output result.mp4
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../checkpoints/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to face image'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to audio file (for simple inference)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.mp4',
        help='Output video path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    
    # STT/TTS options
    parser.add_argument(
        '--use-tts',
        action='store_true',
        help='Use TTS to generate audio from text'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Text to synthesize (with --use-tts)'
    )
    parser.add_argument(
        '--input-audio',
        type=str,
        help='Input audio to transcribe (for conversational mode)'
    )
    parser.add_argument(
        '--response',
        type=str,
        help='Response text to synthesize'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate arguments
    if not args.audio and not args.use_tts and not args.input_audio:
        parser.error("Must provide either --audio, --use-tts with --text, or --input-audio")
    
    if args.use_tts and not args.text and not args.input_audio:
        parser.error("--use-tts requires --text or --input-audio")
    
    # Run inference
    try:
        if args.use_tts or args.input_audio:
            run_with_stt_tts(args, config)
        else:
            run_simple_inference(args, config)
            
    except Exception as e:
        print(f"\n\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

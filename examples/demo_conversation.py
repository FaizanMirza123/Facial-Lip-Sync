"""
Demo script for full conversational AI with lip-sync.
Combines STT, LLM, TTS, and lip-sync in one pipeline.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.realtime_sync import RealtimeLipSync
from stt_tts.tts_integration import STTTTSPipeline


def simple_llm_callback(text: str) -> str:
    """
    Simple LLM callback - replace with your actual LLM integration.
    
    Args:
        text: User input text
        
    Returns:
        LLM response
    """
    # Placeholder: Replace with OpenAI, Llama, or other LLM
    responses = {
        'hello': 'Hello! How can I help you today?',
        'hi': 'Hi there! What can I do for you?',
        'how are you': 'I am doing well, thank you for asking!',
    }
    
    text_lower = text.lower().strip()
    for key in responses:
        if key in text_lower:
            return responses[key]
    
    return f"I understand you said: {text}. How can I help you with that?"


def openai_llm_callback(text: str, api_key: str) -> str:
    """
    OpenAI LLM integration.
    
    Args:
        text: User input
        api_key: OpenAI API key
        
    Returns:
        LLM response
    """
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
    )
    
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(
        description='Conversational AI with Lip-Sync Demo'
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
        help='Path to trained lip-sync model'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to face image'
    )
    parser.add_argument(
        '--input-audio',
        type=str,
        required=True,
        help='Path to user audio input'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='response.mp4',
        help='Output video path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on'
    )
    parser.add_argument(
        '--use-openai',
        action='store_true',
        help='Use OpenAI GPT for LLM (requires OPENAI_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("CONVERSATIONAL AI WITH LIP-SYNC DEMO")
    print("="*70)
    
    # Initialize STT/TTS pipeline
    print("\nInitializing STT/TTS pipeline...")
    stt_tts = STTTTSPipeline(
        stt_config={
            'provider': config['stt_tts']['stt']['provider'],
            'config': {
                'model_name': config['stt_tts']['stt']['model'],
                'language': config['stt_tts']['stt']['language']
            }
        },
        tts_config={
            'provider': config['stt_tts']['tts']['provider'],
            'config': {
                'model_name': config['stt_tts']['tts']['model']
            }
        }
    )
    
    # Setup LLM callback
    if args.use_openai:
        import os
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        llm_callback = lambda text: openai_llm_callback(text, api_key)
        print("Using OpenAI GPT-4")
    else:
        llm_callback = simple_llm_callback
        print("Using simple rule-based responses")
    
    # Process conversation
    print(f"\nProcessing input audio: {args.input_audio}")
    temp_audio = "temp_response_audio.wav"
    
    transcription, llm_response = stt_tts.process_conversation(
        args.input_audio,
        llm_callback,
        temp_audio
    )
    
    print("\n" + "-"*70)
    print(f"USER: {transcription}")
    print(f"AI: {llm_response}")
    print("-"*70)
    
    # Generate lip-synced video
    print("\nGenerating lip-synced video...")
    lip_sync = RealtimeLipSync(
        model_path=args.model,
        config=config,
        device=args.device
    )
    
    print(f"Loading face image: {args.image}")
    lip_sync.set_face_image(args.image)
    
    print(f"Creating video...")
    lip_sync.process_audio_file(temp_audio, args.output)
    
    # Cleanup
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    
    print("\n" + "="*70)
    print(f"SUCCESS! Video saved to: {args.output}")
    print("="*70)
    print("\nThe video shows the AI speaking the response with lip-sync!")


if __name__ == '__main__':
    main()

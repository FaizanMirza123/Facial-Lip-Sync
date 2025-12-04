"""
Text-to-Speech integration.
"""

import torch
import numpy as np
from typing import Optional, Dict
import soundfile as sf
from pathlib import Path


class CoquiTTS:
    """Text-to-Speech using Coqui TTS."""
    
    def __init__(
        self,
        model_name: str = 'tts_models/en/ljspeech/tacotron2-DDC',
        device: str = 'cuda'
    ):
        """
        Initialize Coqui TTS.
        
        Args:
            model_name: TTS model name
            device: Device to run on
        """
        from TTS.api import TTS
        
        self.device = device
        self.tts = TTS(model_name=model_name).to(device)
        
        print(f"Loaded TTS model: {model_name}")
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        speaker: Optional[str] = None
    ):
        """
        Synthesize speech and save to file.
        
        Args:
            text: Input text
            output_path: Output audio file path
            speaker: Speaker name (for multi-speaker models)
        """
        if speaker:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=speaker
            )
        else:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path
            )
    
    def synthesize_to_array(
        self,
        text: str,
        speaker: Optional[str] = None
    ) -> np.ndarray:
        """
        Synthesize speech to numpy array.
        
        Args:
            text: Input text
            speaker: Speaker name
            
        Returns:
            Audio waveform
        """
        if speaker:
            wav = self.tts.tts(text=text, speaker=speaker)
        else:
            wav = self.tts.tts(text=text)
        
        return np.array(wav)
    
    def list_speakers(self) -> list:
        """Get list of available speakers."""
        return self.tts.speakers if hasattr(self.tts, 'speakers') else []


class OpenAITTS:
    """Text-to-Speech using OpenAI API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'tts-1',
        voice: str = 'alloy'
    ):
        """
        Initialize OpenAI TTS.
        
        Args:
            api_key: OpenAI API key
            model: TTS model ('tts-1' or 'tts-1-hd')
            voice: Voice preset ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')
        """
        import openai
        import os
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.voice = voice
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None
    ):
        """
        Synthesize speech and save to file.
        
        Args:
            text: Input text
            output_path: Output audio file path
            voice: Voice preset (overrides default)
        """
        voice = voice or self.voice
        
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text
        )
        
        response.stream_to_file(output_path)
    
    def synthesize_streaming(self, text: str, voice: Optional[str] = None):
        """
        Synthesize speech with streaming.
        
        Args:
            text: Input text
            voice: Voice preset
            
        Returns:
            Audio stream
        """
        voice = voice or self.voice
        
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice,
            input=text
        )
        
        return response.iter_bytes()


class TTSManager:
    """Manager for different TTS providers."""
    
    def __init__(
        self,
        provider: str = 'coqui',
        config: Optional[Dict] = None
    ):
        """
        Initialize TTS manager.
        
        Args:
            provider: TTS provider ('coqui' or 'openai_api')
            config: Provider-specific configuration
        """
        self.provider = provider
        config = config or {}
        
        if provider == 'coqui':
            self.tts = CoquiTTS(**config)
        elif provider == 'openai_api':
            self.tts = OpenAITTS(**config)
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        **kwargs
    ):
        """
        Synthesize speech to file.
        
        Args:
            text: Input text
            output_path: Output audio file path
            **kwargs: Additional provider-specific arguments
        """
        self.tts.synthesize_to_file(text, output_path, **kwargs)


class STTTTSPipeline:
    """Combined STT and TTS pipeline for conversational AI."""
    
    def __init__(
        self,
        stt_config: Dict,
        tts_config: Dict
    ):
        """
        Initialize pipeline.
        
        Args:
            stt_config: STT configuration
            tts_config: TTS configuration
        """
        from stt_tts.whisper_stt import STTManager
        
        self.stt = STTManager(**stt_config)
        self.tts = TTSManager(**tts_config)
    
    def transcribe_and_respond(
        self,
        input_audio_path: str,
        response_text: str,
        output_audio_path: str
    ) -> str:
        """
        Transcribe input audio and generate response audio.
        
        Args:
            input_audio_path: Input audio file
            response_text: Text response to synthesize
            output_audio_path: Output audio file
            
        Returns:
            Transcribed text from input
        """
        # Transcribe input
        transcription = self.stt.transcribe(input_audio_path)
        
        # Generate response
        self.tts.synthesize(response_text, output_audio_path)
        
        return transcription
    
    def process_conversation(
        self,
        audio_path: str,
        llm_callback,
        output_path: str
    ) -> tuple:
        """
        Process full conversation: STT -> LLM -> TTS.
        
        Args:
            audio_path: Input audio
            llm_callback: Function that takes text and returns response
            output_path: Output audio path
            
        Returns:
            Tuple of (transcription, llm_response)
        """
        # Speech to text
        transcription = self.stt.transcribe(audio_path)
        
        # Get LLM response
        llm_response = llm_callback(transcription)
        
        # Text to speech
        self.tts.synthesize(llm_response, output_path)
        
        return transcription, llm_response

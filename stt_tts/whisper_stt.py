"""
Speech-to-Text integration using OpenAI Whisper.
"""

import whisper
import torch
import numpy as np
from typing import Optional, List, Dict
import soundfile as sf


class WhisperSTT:
    """Speech-to-Text using OpenAI Whisper."""
    
    def __init__(
        self,
        model_name: str = 'base',
        device: str = 'cuda',
        language: str = 'en'
    ):
        """
        Initialize Whisper STT.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run on
            language: Language code
        """
        self.device = device
        self.language = language
        
        # Load Whisper model
        self.model = whisper.load_model(model_name, device=device)
        
        print(f"Loaded Whisper {model_name} model on {device}")
    
    def transcribe_file(
        self,
        audio_path: str,
        timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            timestamps: Return word-level timestamps
            
        Returns:
            Dictionary with transcription and metadata
        """
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=timestamps
        )
        
        return result
    
    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Dictionary with transcription
        """
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        result = self.model.transcribe(audio, language=self.language)
        
        return result
    
    def get_word_timings(self, result: Dict) -> List[Dict]:
        """
        Extract word-level timings from transcription.
        
        Args:
            result: Whisper transcription result
            
        Returns:
            List of word timing dictionaries
        """
        word_timings = []
        
        for segment in result['segments']:
            if 'words' in segment:
                for word_info in segment['words']:
                    word_timings.append({
                        'word': word_info['word'],
                        'start': word_info['start'],
                        'end': word_info['end']
                    })
        
        return word_timings
    
    def transcribe_stream(
        self,
        audio_stream,
        chunk_duration: float = 5.0
    ):
        """
        Transcribe real-time audio stream.
        
        Args:
            audio_stream: Audio input stream
            chunk_duration: Duration of chunks to process
        """
        # TODO: Implement streaming transcription
        # This requires buffering audio and processing in chunks
        pass


class OpenAISTT:
    """Speech-to-Text using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI STT.
        
        Args:
            api_key: OpenAI API key (uses env var if None)
        """
        import openai
        import os
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def transcribe_file(
        self,
        audio_path: str,
        language: str = 'en'
    ) -> str:
        """
        Transcribe audio file using OpenAI API.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Transcription text
        """
        with open(audio_path, 'rb') as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language
            )
        
        return transcript.text
    
    def translate_file(
        self,
        audio_path: str
    ) -> str:
        """
        Translate audio to English using OpenAI API.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Translated text
        """
        with open(audio_path, 'rb') as audio_file:
            translation = self.client.audio.translations.create(
                model="whisper-1",
                file=audio_file
            )
        
        return translation.text


class STTManager:
    """Manager for different STT providers."""
    
    def __init__(
        self,
        provider: str = 'whisper',
        config: Optional[Dict] = None
    ):
        """
        Initialize STT manager.
        
        Args:
            provider: STT provider ('whisper' or 'openai_api')
            config: Provider-specific configuration
        """
        self.provider = provider
        config = config or {}
        
        if provider == 'whisper':
            self.stt = WhisperSTT(**config)
        elif provider == 'openai_api':
            self.stt = OpenAISTT(**config)
        else:
            raise ValueError(f"Unknown STT provider: {provider}")
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription text
        """
        if self.provider == 'whisper':
            result = self.stt.transcribe_file(audio_path)
            return result['text']
        else:
            return self.stt.transcribe_file(audio_path)
    
    def get_detailed_transcription(self, audio_path: str) -> Dict:
        """
        Get detailed transcription with timings.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detailed transcription dictionary
        """
        if self.provider == 'whisper':
            return self.stt.transcribe_file(audio_path, timestamps=True)
        else:
            # OpenAI API doesn't provide timestamps
            text = self.stt.transcribe_file(audio_path)
            return {'text': text, 'segments': []}

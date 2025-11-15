"""
Voiceprint Model
Handles voiceprint embeddings and storage
"""

from typing import Dict, Optional
import numpy as np


class VoiceprintModel:
    """
    Manages voiceprint embeddings
    """
    
    def __init__(self):
        self.voiceprints: Dict[str, np.ndarray] = {}
    
    def enroll(self, user_id: str, embedding: np.ndarray):
        """
        Enroll user voiceprint
        """
        self.voiceprints[user_id] = embedding
    
    def get_voiceprint(self, user_id: str) -> Optional[np.ndarray]:
        """
        Get stored voiceprint
        """
        return self.voiceprints.get(user_id)
    
    def compute_embedding(self, audio_data: bytes) -> np.ndarray:
        """
        Compute embedding from audio data
        TODO: Implement actual voice embedding model
        """
        # Mock: return random embedding
        return np.random.rand(128)


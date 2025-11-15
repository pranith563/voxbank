"""
Biometric Matcher
Matches voice samples against enrolled voiceprints
"""

import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity


class BiometricMatcher:
    """
    Matches voice samples using cosine similarity
    """
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def match(self, enrolled_embedding: np.ndarray, sample_embedding: np.ndarray) -> tuple[bool, float]:
        """
        Match voice sample against enrolled voiceprint
        Returns: (is_match, similarity_score)
        """
        if enrolled_embedding is None or sample_embedding is None:
            return False, 0.0
        
        # Reshape for cosine similarity
        enrolled = enrolled_embedding.reshape(1, -1)
        sample = sample_embedding.reshape(1, -1)
        
        similarity = cosine_similarity(enrolled, sample)[0][0]
        is_match = similarity >= self.threshold
        
        return is_match, float(similarity)


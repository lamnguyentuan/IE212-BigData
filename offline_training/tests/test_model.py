"""
Tests for Multimodal Model.
"""
import sys
import torch
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from offline_training.models.multimodal_classifier import MultimodalClassifier

class TestMultimodalModel(unittest.TestCase):
    def test_forward_pass_shapes(self):
        # Setup dims
        v_dim, a_dim, t_dim, m_dim = 768, 768, 768, 22
        fusion_dim = 128
        num_classes = 4
        batch_size = 2
        
        model = MultimodalClassifier(
            video_dim=v_dim,
            audio_dim=a_dim,
            text_dim=t_dim,
            meta_dim=m_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes
        )
        
        # Dummy inputs
        v = torch.randn(batch_size, v_dim)
        a = torch.randn(batch_size, a_dim)
        t = torch.randn(batch_size, t_dim)
        m = torch.randn(batch_size, m_dim)
        
        # Forward
        logits = model(v, a, t, m)
        
        # Assertions
        self.assertEqual(logits.shape, (batch_size, num_classes))
        print("✅ Forward pass shape check passed.")

if __name__ == "__main__":
    unittest.main()

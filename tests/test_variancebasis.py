import pytest
import torch
from variancebasis import VarianceBasis, create_variancebasis

class TestVarianceBasis:
    def test_init(self):
        m = VarianceBasis(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = VarianceBasis(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_variancebasis()
        assert isinstance(m, VarianceBasis)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

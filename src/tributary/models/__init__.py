"""
PyMC model architectures for VOLTA Music Marketing Mix Models.

Three architectures for different assumptions about market similarity:

1. **Pooled**: All markets share identical parameters
   - Assumption: Germany and Poland respond the same to Spotify ads
   - Pro: Maximum data efficiency
   - Con: Ignores real market differences

2. **Unpooled**: Each market has completely independent parameters
   - Assumption: Markets have nothing in common
   - Pro: Maximum flexibility
   - Con: Sparse markets (Poland, Sweden) have very noisy estimates

3. **Hierarchical**: Markets are similar but not identical (partial pooling)
   - Assumption: Markets are "exchangeable" — drawn from common distribution
   - Pro: Sparse markets borrow strength from the group
   - Con: More complex, requires more tuning

The hierarchical model is the star of the show — it's what makes
"structure beat data size" possible.
"""

from tributary.models.pooled import build_pooled_mmm
from tributary.models.unpooled import build_unpooled_mmm
from tributary.models.hierarchical import build_hierarchical_mmm

__all__ = [
    "build_pooled_mmm",
    "build_unpooled_mmm",
    "build_hierarchical_mmm",
]

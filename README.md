# Tributary 🌊

**Can structure beat data size?**

Tributary is a Marketing Mix Modeling (MMM) framework built around hierarchical Bayesian models.  
It demonstrates how *structure* can compensate for limited data by allowing sparse markets to borrow strength from richer ones.

This repository accompanies the **PyMCon 2025** talk:  
**“Hierarchical Models in MMM: Can Structure Beat Data Size?”**

---

## The Problem

You are a data scientist at a music marketing company. An artist is scaling rapidly, and leadership needs ROAS estimates for 8 European markets to plan the next campaign.

The data reality:

| Market | Data Available | Quality |
|------|----------------|---------|
| Germany | 2 years | Strong |
| UK | 2 years | Strong |
| France | 18 months | Medium |
| Netherlands | 1 year | Medium |
| Spain | 1 year | Gaps |
| Italy | 1 year | Medium |
| Poland | 6 months | Sparse |
| Sweden | 6 months | Sparse |

The usual response: *“We need more data.”*  
Tributary’s response: *“We need better structure.”*

---

## The Solution: Partial Pooling

Tributary compares three modeling strategies:

- **Pooled** – All markets share identical parameters  
- **Unpooled** – Each market is modeled independently  
- **Hierarchical (Partial Pooling)** – Markets share information where appropriate

Hierarchical models:
- Stabilize estimates for sparse markets  
- Preserve market-specific signals when data is strong  
- Enable robust decision-making under data imbalance  

---

## The VOLTA Music Group Scenario

VOLTA is a fictional music distribution and marketing company expanding artists across Europe.

**Marketing channels**
- Spotify Ads  
- Meta (Instagram / Facebook)  
- TikTok  
- YouTube Ads  
- Radio Promotion  
- Playlist Pitching  

**Objective**  
Allocate a €500k quarterly budget across markets with unequal data coverage.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/tributary.git
cd tributary
pip install -e ".[dev]"
```

---

### Generate Synthetic Data

```bash
tributary generate
```

This creates:
- `data/volta_marketing.csv`
- `data/ground_truth.json`

---

### Fit Models

```bash
tributary fit data/volta_marketing.csv --model hierarchical
```

Fit all architectures:

```bash
tributary fit data/volta_marketing.csv --model pooled
tributary fit data/volta_marketing.csv --model unpooled
tributary fit data/volta_marketing.csv --model hierarchical
```

Compare models:

```bash
tributary compare results/
```

Run diagnostics:

```bash
tributary evaluate results/hierarchical_trace.nc
```

---

### Notebook Walkthrough

```bash
jupyter lab notebooks/volta_walkthrough.ipynb
```

---

## Repository Structure

```
tributary/
├── src/tributary/
│   ├── transforms/        Adstock and saturation functions
│   ├── models/            Pooled, unpooled, hierarchical MMMs
│   ├── data/              Schemas and synthetic data generator
│   ├── evaluation/        Diagnostics and ROAS analysis
│   └── cli.py             Command-line interface
├── tests/                 Unit and property-based tests
├── notebooks/             End-to-end walkthrough
└── data/                  Generated example data
```

---

## Key Features

### 1. Explicit, Testable Transforms

Media transformations are isolated and testable.

```python
from tributary.transforms import geometric_adstock, hill_saturation

x_adstocked = geometric_adstock(spend, alpha=0.6, l_max=8)
x_saturated = hill_saturation(x_adstocked, K=0.5, S=2.0)
```

All transforms are covered by unit tests and Hypothesis property tests.

---

### 2. Comparable Model Architectures

```python
from tributary.models import (
    build_pooled_mmm,
    build_unpooled_mmm,
    build_hierarchical_mmm,
)
```

Shared priors ensure comparisons isolate *structure*, not tuning.

---

### 3. Decision-Oriented Evaluation

Focuses on business-relevant outputs:

```python
from tributary.evaluation import (
    compute_roas_from_trace,
    compute_shrinkage,
    run_mcmc_diagnostics,
)
```

---

### 4. Strict Data Validation

```python
from tributary.data.schemas import MarketingDataFrame
MarketingDataFrame.validate(df)
```

Errors surface before long MCMC runs.

---

## Shrinkage Intuition

Sparse markets are pulled toward the group mean.  
Data-rich markets retain their individuality.

This stabilizes ROAS estimates where data is thin, without over-smoothing where data is strong.

---

## When Hierarchical Models Fail

Partial pooling is not a silver bullet. It performs poorly when:
1. Markets are fundamentally unrelated  
2. All groups are extremely data-poor  
3. The hierarchy is mis-specified  

The notebooks include failure-mode examples.

---

## Requirements

- Python ≥ 3.10  
- PyMC ≥ 5.10  
- ArviZ ≥ 0.17  
- pandas, numpy, pydantic, pandera, typer  

See `pyproject.toml` for full dependencies.

---

## Tests

```bash
pytest
pytest -m "not slow"
pytest --cov=tributary --cov-report=html
```

---

## Talk Resources

- Slides: *TBD*  
- Recording: *TBD*  
- Blog post: *TBD*  

---

## Citation

```bibtex
@software{tributary2025,
  author = {Your Name},
  title = {Tributary: Hierarchical Marketing Mix Models with PyMC},
  year = {2025},
  url = {https://github.com/yourusername/tributary}
}
```

---

## License

MIT License.

---

**Structure can beat volume — when used deliberately. 🌊**

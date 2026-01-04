# Tributary 🌊

**Can structure beat data size?**

Hierarchical Marketing Mix Models that let sparse markets borrow strength from the group.

Built for **PyMCon 2025**: *"Hierarchical Models in MMM: Can Structure Beat Data Size?"*

---

## The Problem

You're a data scientist at a music marketing company. Your artist is blowing up, and leadership wants ROAS estimates for 8 European markets to plan the next campaign.

But here's the reality:

| Market | Data Available | Quality |
|--------|----------------|---------|
| 🇩🇪 Germany | 2 years | Good |
| 🇬🇧 UK | 2 years | Good |
| 🇫🇷 France | 18 months | OK |
| 🇳🇱 Netherlands | 1 year | OK |
| 🇪🇸 Spain | 1 year | Gaps in TikTok |
| 🇮🇹 Italy | 1 year | OK |
| 🇵🇱 Poland | 6 months | Sparse! |
| 🇸🇪 Sweden | 6 months | Sparse! |

The usual answer: *"We need more data."*

**Tributary's answer:** *"We need better structure."*

---

## The Solution: Partial Pooling

Instead of:
- **Pooled**: Pretending all markets are identical (too rigid)
- **Unpooled**: Treating each market as completely independent (too noisy for sparse markets)

We use **hierarchical models** with partial pooling:
- Markets with thin data *borrow strength* from the group
- Markets with strong signals *pull away* from the mean
- You get stability where you need it, flexibility where the data supports it

---

## The VOLTA Music Group Scenario

**VOLTA** is a music distribution and marketing company helping independent artists break into European markets.

**Channels:**
- 🎧 Spotify Ads
- 📱 Meta (Instagram/Facebook)
- 🎵 TikTok
- 📺 YouTube Ads
- 📻 Radio Promotion
- 🎼 Playlist Pitching

**Challenge:** Allocate €500K quarterly budget across markets with wildly different data availability.

---

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/tributary.git
cd tributary

"""Static business descriptions for pair-selection and direction prompts.

Undated annual revenue figures are deliberately excluded: without release-time
metadata, even a fiscal year preceding a decision can contain unavailable facts.
"""
from __future__ import annotations

from typing import Dict, Sequence

COMPANY_PROFILES: Dict[str, Dict[str, str]] = {
    "AAPL": {"name": "Apple Inc.",
             "business": "Consumer electronics (iPhone, Mac, iPad, Watch) and high-margin Services (App Store, iCloud, payments)."},
    "ADBE": {"name": "Adobe Inc.",
             "business": "Subscription creative & document software (Creative Cloud, Document Cloud) and digital-experience/marketing software."},
    "AMZN": {"name": "Amazon.com, Inc.",
             "business": "E-commerce marketplace and logistics, AWS cloud computing, and advertising."},
    "BMRN": {"name": "BioMarin Pharmaceutical Inc.",
             "business": "Biotech developing enzyme-replacement and gene therapies for rare genetic diseases."},
    "CRM": {"name": "Salesforce, Inc.",
            "business": "Cloud-based CRM and enterprise SaaS (sales, service, marketing, analytics, Slack)."},
    "GOOGL": {"name": "Alphabet Inc. (Google)",
              "business": "Search and digital advertising, YouTube, Google Cloud, and Android/Play."},
    "META": {"name": "Meta Platforms, Inc.",
             "business": "Social-media advertising (Facebook, Instagram, WhatsApp) and Reality Labs (AR/VR)."},
    "MSFT": {"name": "Microsoft Corporation",
             "business": "Productivity software (Office/365), Azure cloud, Windows, and gaming (Xbox)."},
    "NVDA": {"name": "NVIDIA Corporation",
             "business": "GPUs and AI accelerators; data-center compute, gaming, and professional visualization."},
    "TSLA": {"name": "Tesla, Inc.",
             "business": "Electric vehicles, energy storage/solar, and related software/services."},
}


def profile_text(ticker: str) -> str:
    """One-line business context; no unversioned financial results reach the LLM."""
    p = COMPANY_PROFILES.get(str(ticker).upper())
    if not p:
        return f"{ticker}: (no company profile on file)"
    return f"{ticker} — {p['name']}: {p['business']}"


def profiles_block(tickers: Sequence[str]) -> str:
    """Render stable business context without annual financial-result leakage."""
    return "COMPANY CONTEXT (business only):\n" + "\n".join(
        f"- {profile_text(t)}" for t in tickers)

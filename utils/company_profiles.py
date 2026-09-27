"""Static company business + 2024 revenue context for the trading universe (Step 5).

SOURCE & CAVEAT
---------------
These are APPROXIMATE figures compiled from each company's public annual report / Form
10-K (fiscal year noted per company; several tech firms do not use a calendar fiscal
year). They are rounded to the nearest ~$0.1B and are intended as qualitative business
context for the LLM, NOT as precise accounting inputs. **Verify against the official
filing before any publication.** No values here are fabricated; where a fiscal year does
not align to calendar 2024 the basis is stated explicitly.

Each entry: ticker -> {name, business, revenue_2024 (text, with FY basis)}.
"""
from __future__ import annotations

from typing import Dict, Sequence

COMPANY_PROFILES: Dict[str, Dict[str, str]] = {
    "AAPL": {"name": "Apple Inc.",
             "business": "Consumer electronics (iPhone, Mac, iPad, Watch) and high-margin Services (App Store, iCloud, payments).",
             "revenue_2024": "~$391B (FY2024, ended Sep 2024)"},
    "ADBE": {"name": "Adobe Inc.",
             "business": "Subscription creative & document software (Creative Cloud, Document Cloud) and digital-experience/marketing software.",
             "revenue_2024": "~$21.5B (FY2024, ended Nov 2024)"},
    "AMZN": {"name": "Amazon.com, Inc.",
             "business": "E-commerce marketplace and logistics, AWS cloud computing, and advertising.",
             "revenue_2024": "~$638B (FY2024, calendar)"},
    "BMRN": {"name": "BioMarin Pharmaceutical Inc.",
             "business": "Biotech developing enzyme-replacement and gene therapies for rare genetic diseases.",
             "revenue_2024": "~$2.85B (FY2024, calendar)"},
    "CRM": {"name": "Salesforce, Inc.",
            "business": "Cloud-based CRM and enterprise SaaS (sales, service, marketing, analytics, Slack).",
            "revenue_2024": "~$34.9B (FY2024, ended Jan 2024)"},
    "GOOGL": {"name": "Alphabet Inc. (Google)",
              "business": "Search and digital advertising, YouTube, Google Cloud, and Android/Play.",
              "revenue_2024": "~$350B (FY2024, calendar)"},
    "META": {"name": "Meta Platforms, Inc.",
             "business": "Social-media advertising (Facebook, Instagram, WhatsApp) and Reality Labs (AR/VR).",
             "revenue_2024": "~$164.5B (FY2024, calendar)"},
    "MSFT": {"name": "Microsoft Corporation",
             "business": "Productivity software (Office/365), Azure cloud, Windows, and gaming (Xbox).",
             "revenue_2024": "~$245B (FY2024, ended Jun 2024)"},
    "NVDA": {"name": "NVIDIA Corporation",
             "business": "GPUs and AI accelerators; data-center compute, gaming, and professional visualization.",
             "revenue_2024": "~$60.9B (FY2024, ended Jan 2024); ~$130.5B (FY2025, ended Jan 2025)"},
    "TSLA": {"name": "Tesla, Inc.",
             "business": "Electric vehicles, energy storage/solar, and related software/services.",
             "revenue_2024": "~$97.7B (FY2024, calendar)"},
}


def profile_text(ticker: str) -> str:
    """One-line business + 2024-revenue context for a ticker (or a neutral fallback)."""
    p = COMPANY_PROFILES.get(str(ticker).upper())
    if not p:
        return f"{ticker}: (no company profile on file)"
    return f"{ticker} — {p['name']}: {p['business']} 2024 revenue {p['revenue_2024']}."


def profiles_block(tickers: Sequence[str]) -> str:
    """A 'COMPANY CONTEXT' block for the given tickers, for prompt injection."""
    lines = [profile_text(t) for t in tickers]
    return "COMPANY CONTEXT (business + 2024 revenue):\n" + "\n".join(f"- {l}" for l in lines)

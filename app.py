import streamlit as st
import csv
import json
from genomics_interpreter import (
    TRAIT_DB_PATH,
    load_trait_database,
    parse_genotype_file,
    match_traits,
    build_report_object,
    generate_ai_summary,
    generate_text_report,
    generate_html_report,
)
from openai import OpenAI

client = OpenAI()  # uses your OPENAI_API_KEY

# ---------- Page config ----------
st.set_page_config(
    page_title="GenAI Engine",
    page_icon="🧬",
    layout="wide",
)

# ---------- Global styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg-dark: #020617;
        --bg-main: #020617;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-gold: #fbbf24;
        --text-muted: #9ca3af;
        --text-main: #e5e7eb;
    }
    body {
        background: var(--bg-main);
        color: var(--text-main);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--bg-main);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .main .block-container {
        padding-top: 0;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* TOP NAV */

    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 0 10px;
        margin-bottom: 6px;
    }

    .top-nav-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* G shaped DNA style mark */

    .logo-mark {
        width: 32px;
        height: 32px;
        border-radius: 999px;
        background: #020617;
        border: 2px solid #3b82f6;
        position: relative;
        box-shadow: 0 6px 14px rgba(15,23,42,0.45);
    }

    .logo-g-arc {
        position: absolute;
        inset: 4px;
        border-radius: 999px;
        border: 2px solid #3b82f6;
        border-right-color: transparent;
    }

    .logo-g-cut {
        position: absolute;
        right: 4px;
        top: 50%;
        width: 8px;
        height: 2px;
        background: #020617;
        transform: translateY(-50%);
    }

    .logo-helix-line {
        position: absolute;
        top: 6px;
        bottom: 6px;
        left: 50%;
        width: 2px;
        border-radius: 999px;
        transform: translateX(-7px);
        background: linear-gradient(180deg, #3b82f6, #22c55e);
    }

    .logo-helix-line:nth-child(2) {
        transform: translateX(5px);
        background: linear-gradient(180deg, #8b5cf6, #fbbf24);
    }

    .logo-rung {
        position: absolute;
        left: 50%;
        width: 14px;
        height: 2px;
        background: #e5e7eb;
        border-radius: 999px;
        transform: translateX(-50%);
    }

    .logo-rung:nth-child(1) { top: 9px; }
    .logo-rung:nth-child(2) { top: 16px; }
    .logo-rung:nth-child(3) { top: 23px; }

    .logo-text-main {
        font-weight: 700;
        letter-spacing: 0.12em;
        font-size: 0.86rem;
        text-transform: uppercase;
    }

    .logo-text-sub {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: -2px;
    }

    /* Nav buttons (top right) */

    .nav-container {
        display: flex;
        justify-content: flex-end;
        gap: 18px;
        padding-top: 4px;
    }

    .stButton>button {
        background: transparent;
        border: none;
        color: #4b5563;
        padding: 2px 0;
        font-size: 0.9rem;
        font-weight: 500;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        transition: color 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
    }

    .stButton>button:hover {
        color: #111827;
        border-bottom-color: #d1d5db;
        transform: translateY(-1px);
        cursor: pointer;
    }

    .nav-active > button {
        border-bottom-color: #3b82f6 !important;
        color: #111827 !important;
    }

    /* HERO BAND */

    .hero-band {
        background: radial-gradient(circle at 0% 0%, #0f172a, #020617 48%);
        color: #e5e7eb;
        border-radius: 0 0 40px 40px;
        padding: 40px 56px 64px;
        margin: 0 -3rem 40px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.55);
    }

    @media (max-width: 800px) {
        .hero-band {
            margin: 0 -1rem 32px;
            padding: 28px 24px 44px;
        }
    }

    .hero-grid {
        display: flex;
        gap: 56px;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }

    .hero-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.5);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 12px;
        color: #e5e7eb;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }

    .hero-title span {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
        -webkit-background-clip: text;
        color: transparent;
    }

    .hero-sub {
        font-size: 0.98rem;
        max-width: 560px;
        color: #cbd5f5;
        margin-bottom: 1.0rem;
    }

    .hero-note {
        font-size: 0.8rem;
        color: #9ca3af;
        max-width: 540px;
    }

    /* DNA CARD (G-Helix inspired) */

    .hero-dna-card {
        min-width: 260px;
        max-width: 320px;
        border-radius: 24px;
        background:
            radial-gradient(circle at 10% 0%, #1f2937, transparent 55%),
            radial-gradient(circle at 90% 100%, rgba(251,191,36,0.25), transparent 50%),
            #020617;
        border: 1px solid rgba(148, 163, 184, 0.45);
        padding: 18px 18px 20px;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }

    .hero-dna-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 30px rgba(15,23,42,0.7);
        border-color: rgba(251,191,36,0.8);
    }

    .hero-dna-title {
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .hero-dna-sub {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 12px;
    }

    .dna-helix {
        position: relative;
        height: 180px;
        margin-top: 4px;
    }

    .dna-strand {
        position: absolute;
        width: 2px;
        height: 100%;
        border-radius: 999px;
        left: 50%;
        transform: translateX(-26px);
        background: linear-gradient(180deg, #3b82f6, #f97316);
    }

    .dna-strand:nth-child(2) {
        transform: translateX(26px);
        background: linear-gradient(180deg, #8b5cf6, #22c55e);
    }

    .dna-rung {
        position: absolute;
        width: 64px;
        height: 2px;
        background: rgba(148, 163, 184, 0.7);
        left: 50%;
        transform: translateX(-50%);
        border-radius: 999px;
    }

    .dna-rung:nth-child(3) { top: 12%; }
    .dna-rung:nth-child(4) { top: 28%; }
    .dna-rung:nth-child(5) { top: 44%; }
    .dna-rung:nth-child(6) { top: 60%; }
    .dna-rung:nth-child(7) { top: 76%; }

    /* GENERAL SECTIONS */

    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        color: var(--text-main);
    }

    .section-sub {
        font-size: 0.94rem;
        color: #6b7280;
        margin-bottom: 1.0rem;
    }

    .home-section {
        margin-top: 0.5rem;
        margin-bottom: 2.5rem;
    }

    .newsletter-modal {
        border-radius: 16px;
        padding: 14px 16px 12px;
        border: 1px solid #e5e7eb;
        background: #f9fafb;
        box-shadow: 0 20px 40px rgba(15,23,42,0.12);
        color: var(--text-main);
    }

    .newsletter-title {
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    .newsletter-sub {
        font-size: 0.86rem;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }

    .small-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    /* FEATURE / INFO CARDS */

    .feature-row {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .feature-card {
        flex: 1 1 220px;
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        padding: 14px 16px 16px;
        box-shadow: 0 8px 18px rgba(148, 163, 184, 0.18);
        color: var(--text-main);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(148,163,184,0.35);
        border-color: #cbd5f5;
    }

/* FIX HOW-SECTION TITLES */
.how-title,
.how-label,
.how-title *,
.how-label * {
    color: #111827 !important;
}

/* FIX NEWSLETTER SECTION TEXT */
.newsletter-title,
.newsletter-title *,
.newsletter-sub,
.newsletter-sub * {
    color: #111827 !important;
}

    .feature-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }

    .feature-title {
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        color: var(--text-main);
    }

    .feature-body {
        font-size: 0.86rem;
        color: #4b5563;
    }

/* FORCE DARK TEXT INSIDE LIGHT CARDS */
.feature-card,
.feature-card * {
    color: #111827 !important;
}

.how-step,
.how-step * {
    color: #111827 !important;
}

.bio-card,
.bio-card * {
    color: #111827 !important;
}

.who-card,
.who-card * {
    color: #111827 !important;
}
    /* HOW IT WORKS STRIP */

    .how-band {
        margin-top: 2.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #f9fafb, #eef2ff);
        border: 1px solid #e0e7ff;
        padding: 16px 18px 18px;
        color: var(--text-main);
    }

    .how-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b7280;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }

    .how-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .how-row {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
    }

    .how-step {
        flex: 1 1 190px;
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7ff;
        padding: 10px 12px 12px;
        font-size: 0.84rem;
        color: #4b5563;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .how-step:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 22px rgba(148,163,184,0.35);
        border-color: #c7d2fe;
    }

    .how-step-number {
        font-size: 0.75rem;
        font-weight: 700;
        color: #6366f1;
        margin-bottom: 0.1rem;
    }

    .how-step-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        color: var(--text-main);
    }

    /* FOUNDER + WHO IT'S FOR + ROADMAP */

    .bio-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        padding: 14px 16px 16px;
        font-size: 0.9rem;
        box-shadow: 0 8px 18px rgba(148, 163, 184, 0.18);
        color: var(--text-main);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .bio-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(148,163,184,0.35);
        border-color: #cbd5f5;
    }

    .bio-name {
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: var(--text-main);
    }

    .bio-role {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 0.6rem;
    }

    .who-row {
        display: flex;
        gap: 18px;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }

    .who-card {
        flex: 1 1 220px;
        background: #f9fafb;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        padding: 10px 12px 12px;
        font-size: 0.86rem;
        color: #4b5563;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .who-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 22px rgba(148,163,184,0.35);
        border-color: #d1d5db;
    }

    .who-title {
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: var(--text-main);
    }

    .roadmap-card {
        background: #020617;
        color: #e5e7eb;
        border-radius: 18px;
        padding: 14px 16px 16px;
        border: 1px solid rgba(148,163,184,0.5);
        font-size: 0.86rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }

    .roadmap-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 32px rgba(15,23,42,0.8);
        border-color: rgba(59,130,246,0.7);
    }

    .roadmap-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9ca3af;
        margin-bottom: 0.2rem;
    }

    .roadmap-title {
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    /* FINAL FIX: Ensure How-Title is dark */
    .how-title {
        color: #111827 !important;
    }
/* Global site footer */
.site-footer {
    margin-top: 40px;
    padding: 18px 0 10px;
    border-top: 1px solid rgba(148, 163, 184, 0.4);
    font-size: 0.8rem;
    color: #9ca3af;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    opacity: 0.9;
}
.site-footer a {
    color: #9ca3af;
    text-decoration: none;
}
.site-footer a:hover {
    text-decoration: underline;
    color: #e5e7eb;
}

/* Smooth section fade-in */
.hero,
.home-section,
.section-light,
.how-band,
.chat-box,
.feature-card {
    animation: fadeUp 0.35s ease-out;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Button hover transitions */
.stButton > button {
    transition: background-color 0.18s ease, color 0.18s ease,
                transform 0.18s ease, box-shadow 0.18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.35);
}

/* Card hover transitions */
.feature-card,
.how-step,
.trait-card,
.chat-box {
    transition: transform 0.18s ease-out, box-shadow 0.18s ease-out,
                border-color 0.18s ease-out, background-color 0.18s ease-out;
}

.feature-card:hover,
.how-step:hover,
.trait-card:hover,
.chat-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.4);
    border-color: rgba(148, 163, 184, 0.7);
}

/* Navbar hover refinement */
.nav-btn,
.nav-btn-active {
    transition: color 0.16s ease, border-bottom-color 0.16s ease,
                transform 0.16s ease;
}

.nav-btn:hover,
.nav-btn-active:hover {
    transform: translateY(-1px);
}

/* ---------- Microanimations + Page Transitions ---------- */

/* Page fade / slide on navigation (Streamlit rerun, but looks like a transition) */
.page-wrap {
  animation: pageFade 240ms ease-out;
}

@keyframes pageFade {
  from { opacity: 0; transform: translateY(6px); filter: blur(1px); }
  to   { opacity: 1; transform: translateY(0); filter: blur(0); }
}

/* Premium button microinteractions */
.stButton > button {
  transition: transform 180ms ease, box-shadow 180ms ease, background-color 180ms ease, color 180ms ease;
  will-change: transform;
}

.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 26px rgba(15, 23, 42, 0.45);
}

.stButton > button:active {
  transform: translateY(0px) scale(0.98);
  box-shadow: 0 6px 14px rgba(15, 23, 42, 0.35);
}

/* Card + panel hover polish */
.feature-card, .how-step, .trait-card, .chat-box, .section-light {
  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
  will-change: transform;
}

.feature-card:hover, .how-step:hover, .trait-card:hover, .chat-box:hover, .section-light:hover {
  transform: translateY(-3px);
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.45);
  border-color: rgba(148, 163, 184, 0.75);
}

/* Navbar microinteraction (bolder + subtle lift) */
.nav-btn {
  font-weight: 600 !important;
  letter-spacing: 0.2px;
  transition: color 160ms ease, border-bottom-color 160ms ease, transform 160ms ease;
}

.nav-btn:hover {
  transform: translateY(-1px);
}

html {
  scroll-behavior: smooth;
}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Nav state ----------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

# Page transition key (increments on navigation change)
if "page_transition_key" not in st.session_state:
    st.session_state.page_transition_key = 0


def set_page(page_name: str):
    st.session_state.active_page = page_name
    st.session_state.page_transition_key += 1


page = st.session_state.active_page

# Wrapper to trigger CSS animation on page change
st.markdown(
    f"<div class='page-wrap' data-k='{st.session_state.page_transition_key}'>",
    unsafe_allow_html=True,
)

# ---------- TOP NAV BAR ----------
with st.container():
    col_left, col_right = st.columns([1.4, 1.6])

    with col_left:
        st.markdown(
            """
            <style>
            .nav-logo-container {
                display: flex;
                align-items: center;
                gap: 14px;
            }
            .logo-title {
                font-weight: 700;
                letter-spacing: 0.12em;
                font-size: 1.05rem;
                text-transform: uppercase;
            }
            .logo-sub {
                font-size: 0.75rem;
                color: #9ca3af;
                margin-top: -4px;
            }
            .nav-button-container {
                display: flex;
                justify-content: flex-end;
                gap: 20px;
                padding-top: 8px;
            }
            .nav-btn {
    background: transparent !important;
    border: none !important;
    font-size: 0.95rem !important;
    color: #e5e7eb !important;              /* light text on dark nav */
    border-bottom: 2px solid transparent !important;
    padding-bottom: 4px !important;
}

.nav-btn:hover {
    color: #ffffff !important;              /* bright white on hover */
    border-bottom-color: #38bdf8 !important; /* cyan underline */
    transform: translateY(-1px);
}

.nav-btn-active {
    background: transparent !important;
    color: #ffffff !important;              /* active tab always bright */
    border-bottom: 2px solid #3b82f6 !important; /* blue underline */
    padding-bottom: 4px !important;
transition: color 0.16s ease, border-bottom-color 0.16s ease,
                transform 0.16s ease;
}

/* G Helix rotating DNA mark */
.g-helix-orbit {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    position: relative;
    border: 1px solid rgba(148, 163, 184, 0.6);
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: helix-rotate 4.5s linear infinite;
    background: radial-gradient(circle at 30% 30%, rgba(56, 189, 248, 0.3), transparent 55%);
}

.g-helix-strand {
    position: absolute;
    width: 2px;
    height: 80%;
    border-radius: 999px;
    background: linear-gradient(to bottom, #38bdf8, #a855f7);
    opacity: 0.95;
}

.g-helix-strand-left {
    transform: rotate(18deg) translateX(-7px);
}

.g-helix-strand-right {
    transform: rotate(-18deg) translateX(7px);
}

.g-helix-rung {
    position: absolute;
    width: 26px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(to right, #38bdf8, #a855f7);
    opacity: 0.9;
}

.g-helix-rung.rung-1 { top: 18%; }
.g-helix-rung.rung-2 { top: 35%; }
.g-helix-rung.rung-3 { top: 55%; }
.g-helix-rung.rung-4 { top: 72%; }

@keyframes helix-rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.image("genai_logo.png", width=58)

        st.markdown(
            """
            <div class="nav-logo-container">
                <div>
                    <div class="logo-title">GENAI ENGINE</div>
                    <div class="logo-sub">AI powered genomic insight</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("<div class='nav-button-container'>", unsafe_allow_html=True)
        nav_cols = st.columns(7)

        pages = ["Home", "Upload & Report", "Lifestyle Chatbot", "Trait Explorer", "Trait Science", "About", "Contact"]
        labels = ["Home", "Upload", "Lifestyle chatbot", "Trait explorer", "Trait science", "About", "Contact"]

        for i, p in enumerate(pages):
            active = st.session_state.get("active_page", "Home") == p
            btn_classes = "nav-btn nav-btn-active" if active else "nav-btn"
            # Streamlit does not let us set classes per button directly, so we just use the label
            if nav_cols[i].button(labels[i], key=f"nav_{p}"):
                st.session_state.active_page = p

        st.markdown("</div>", unsafe_allow_html=True)
              
           
# ---------- Newsletter state ----------
if "hide_newsletter" not in st.session_state:
    st.session_state.hide_newsletter = False


def newsletter_block(location_text: str):
    if st.session_state.hide_newsletter:
        return

    with st.container():
        st.markdown(
            f"<div class='small-label'>Stay in the loop</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='newsletter-modal'>"
            "<div class='newsletter-title'>Join the GenAI Engine early access list</div>"
            "<div class='newsletter-sub'>Get updates as the platform evolves and, in a full launch, "
            "receive your reports securely by email.</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns([3, 1])
        with cols[0]:
            email = st.text_input("Email address", key=f"newsletter_{location_text}")
        with cols[1]:
            st.write("")
            if st.button("Notify me", key=f"notify_{location_text}"):
                if email.strip():
                    st.success("Thanks for subscribing. In a real deployment this would save to a mailing list.")
                    st.session_state.hide_newsletter = True
                else:
                    st.warning("Please enter an email.")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Trait DB helper ----------
def load_trait_rows_from_csv(path: str):
    """Load the raw trait database as a list of row dicts for exploration."""
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        st.error(f"Could not load trait database from {path}: {e}")
    return rows

# ---------- HOME ----------
if page == "Home":
    st.markdown(
        """
        <div class="hero-band">
          <div class="hero-grid">
            <div>
              <div class="hero-chip">GenAI Engine · Prototype</div>
              <div class="hero-title">
                Genetics is shaping the future.<br/>
                Understand your genome with <span>AI powered insight.</span>
              </div>
              <div class="hero-sub">
                The way your body responds to sleep, stress, food, exercise, and focus is influenced in part by your DNA.
                As sequencing becomes more common, genetics is becoming a real force in healthcare and everyday life.
                GenAI Engine exists to make those signals easier to understand without treating DNA as destiny. The goal
                is to help you see how genetics may interact with your lifestyle so you can think about realistic ways
                to support your long term health and performance.
              </div>
              <div class="hero-note">
                This prototype runs locally, uses a small curated trait panel, and is meant for education and self reflection,
                not for diagnosis or treatment.
              </div>
            </div>
            <div class="hero-dna-card">
              <div class="hero-dna-title">The G Helix engine.</div>
              <div class="hero-dna-sub">
                A lightweight variant interpreter feeds a language model that explains patterns
                in clear language while keeping the underlying science visible.
              </div>
              <div class="dna-helix">
                <div class="dna-strand"></div>
                <div class="dna-strand"></div>
                <div class="dna-rung"></div>
                <div class="dna-rung"></div>
                <div class="dna-rung"></div>
                <div class="dna-rung"></div>
                <div class="dna-rung"></div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Primary CTAs under hero
    cta_col1, cta_col2 = st.columns([0.4, 0.4])
    with cta_col1:
        if st.button("Generate a demo report", key="hero_demo_report"):
            st.session_state.active_page = "Upload & Report"
    with cta_col2:
        if st.button("Try the lifestyle chatbot", key="hero_chatbot"):
            st.session_state.active_page = "Lifestyle Chatbot"

    # Why GenAI Engine exists
    st.markdown('<div class="home-section dark-section">', unsafe_allow_html=True)
    col_left, col_right = st.columns([1.4, 1.0])

    with col_left:
        st.markdown('<div class="section-title">Why GenAI Engine exists</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - Turn raw SNP data into something a person can actually read.  
            - Help people see how certain traits may connect to sleep, nutrition, focus, or training response.  
            - Give students and future clinicians a safe way to practice genomic thinking without making medical claims.  
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        newsletter_block("home")
    st.markdown("</div>", unsafe_allow_html=True)

    # Why choose GenAI Engine
    st.markdown('<div class="home-section dark-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why choose GenAI Engine?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A focused, lifestyle aware, education first platform instead of a generic consumer report.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-row">
          <div class="feature-card">
            <div class="feature-label">Founder story</div>
            <div class="feature-title">Designed by a future AI geneticist</div>
            <div class="feature-body">
              GenAI Engine is led by Anjali, a high school student who plans to work at the intersection of genetics,
              AI, and clinical decision making. The project reflects real experience in rare disease research and
              clinical volunteering, translated into a tool that explains DNA in a calm and structured way.
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-label">Transparent</div>
            <div class="feature-title">Every trait is traceable</div>
            <div class="feature-body">
              Under the hood is a curated trait table. Each interpretation links to specific rsIDs and genes,
              rather than opaque risk scores, so you can always see the structure behind the narrative.
            </div>
          </div>
          <div class="feature-card">
            <div class="feature-label">Lifestyle focus</div>
            <div class="feature-title">DNA as one piece of the puzzle</div>
            <div class="feature-body">
              The goal is not to label you. It is to highlight patterns that might interact with sleep, nutrition,
              attention, or training, always framed as possibilities and ideas, never as prescriptions.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="home-section">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="how-band">
          <div class="how-label">How it works</div>
          <div class="how-title">From raw file to a lifestyle aware report.</div>
          <div class="how-row">
            <div class="how-step">
              <div class="how-step-number">01</div>
              <div class="how-step-title">Upload a raw DNA file</div>
              <div>Use a 23andMe style text export or start with the built in demo file to see the flow.</div>
            </div>
            <div class="how-step">
              <div class="how-step-number">02</div>
              <div class="how-step-title">Match variants to traits</div>
              <div>A genetics engine scans rsIDs and matches them to a curated panel related to traits such as caffeine response, sleep, or recovery.</div>
            </div>
            <div class="how-step">
              <div class="how-step-number">03</div>
              <div class="how-step-title">Generate an AI overview</div>
              <div>The language model converts structured traits into a human readable summary that focuses on education and realistic possibilities.</div>
            </div>
            <div class="how-step">
              <div class="how-step-number">04</div>
              <div class="how-step-title">Review the full report</div>
              <div>Explore a printable HTML report with trait cards, explanations, and caveats that you can revisit over time.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Secondary CTA near How it works
    spacer_left, cta_mid, spacer_right = st.columns([0.35, 0.3, 0.35])
    with cta_mid:
        if st.button("Start with a demo report", key="how_demo_report"):
            st.session_state.active_page = "Upload & Report"

    # Founder profile / Who this is for / Roadmap
    st.markdown('<div class="home-section dark-section">', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1.1])

    with col_a:
        st.markdown('<div class="section-title">Founder profile</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="bio-card">
              <div class="bio-name">Anjali Nalla</div>
              <div class="bio-role">Student and aspiring AI geneticist</div>
              <div>
                I'm Anjali Nalla, a high school student with hands-on experience in genetics, neuroscience, and clinical settings.
                My background includes Johns Hopkins affiliated research focused on rare disease biology, internships that explore
                both medicine and research, and formal laboratory certifications that trained me in careful, real-world lab practice.
                I am especially interested in genetic counseling and neuro oncology, and in how AI can support thoughtful,
                evidence-aware conversations about risk and lifestyle. GenAI Engine is my way of connecting everything I am
                learning into a single pipeline: from raw genomic data, to structured trait interpretation, to clear explanations
                that help people understand their biology in a calm and responsible way.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown('<div class="section-title">Who this is for</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="who-row">
              <div class="who-card">
                <div class="who-title">People curious about DNA and lifestyle</div>
                <div>You want to understand how genetics may relate to your sleep, focus, training, or nutrition
                without expecting a diagnosis or a quick fix.</div>
              </div>
              <div class="who-card">
                <div class="who-title">Students in biology or pre medicine</div>
                <div>You would like a sandbox where you can practice reading trait reports, understand limitations,
                and think about how you would communicate genomics in the future.</div>
              </div>
              <div class="who-card">
                <div class="who-title">Future clinicians and AI builders</div>
                <div>You are interested in combining structured genomic data, language models, and human centered design
                in a responsible way.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown('<div class="section-title">Roadmap</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="roadmap-card">
              <div class="roadmap-label">Next steps</div>
              <div class="roadmap-title">Beyond this prototype</div>
              <div>
                · Expanding the trait panel with additional genotypes and pathways that relate to lifestyle. <br/>
                · Account creation so users can save reports under a GenAI Engine login. <br/>
                · Email delivery of reports from a dedicated GenAI Engine address. <br/>
                · A cautious chatbot that lets users ask follow up questions about their traits and habits, framed as education
                  and ideas rather than medical advice.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div style="margin-top: 0.8rem; font-size: 0.86rem; color: #e5e7eb;">
          <strong>Founder highlights</strong><br/>
          · Johns Hopkins affiliated research in rare disease biology<br/>
          · Clinical volunteering on a neuro floor and in hospice settings<br/>
          · Internships that combine medicine, psychology, and research<br/>
          · Formal lab safety and benchwork certifications<br/>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Safety, privacy, and limitations (Home)
    st.markdown('<div class="home-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Safety, privacy, and limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - GenAI Engine is an educational prototype. It does not diagnose, treat, or predict disease.  
        - This tool focuses on a small set of lifestyle-related traits, not your full genomic risk.  
        - In this local version, files are processed for the session and not saved to a user database.  
        - Genetics is only one part of the picture alongside sleep, nutrition, stress, and environment.  
        - For any medical questions or concerns, always speak with a licensed clinician or genetic counselor.  
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # FAQ (Home - short)
    st.markdown('<div class="home-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick questions</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **What kind of DNA files can I use?**  
        Plain text genotype files with columns like rsid, chromosome, position, and genotype. A 23andMe-style export is a common example.  

        **Will this tell me if I have a disease?**  
        No. GenAI Engine only looks at a small set of lifestyle-oriented traits and does not provide medical risk predictions.  

        **Is this a replacement for my doctor or genetic counselor?**  
        No. This is a learning tool. It can help you think of questions, but it cannot replace professional advice.  
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Upload & Report":
    st.markdown('<div class="section-title">Generate a personal trait report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Upload a compatible raw DNA text file. GenAI Engine will interpret a subset of variants, build a trait '
        'profile, and ask an AI model to generate a plain language overview with a focus on traits that may interact with lifestyle.</div>',
        unsafe_allow_html=True,
    )

    newsletter_block("upload")

    col1, col2 = st.columns([1.2, 0.9])

    with col1:
        st.markdown("#### Upload raw data")
        uploaded = st.file_uploader("Raw genotype file (.txt, 23andMe style)", type=["txt"])

        st.caption(
            "The file should contain at least rsid and genotype columns. "
            "For testing, you can use the built in demo file."
        )

        use_demo = st.checkbox("Use demo file from this project (test_genotype.txt)", value=not bool(uploaded))

        st.markdown("#### Where should results go?")
        email_for_result = st.text_input(
            "Optional email (for a future version that emails the PDF)",
            placeholder="you@example.com",
        )

        generate = st.button("Run analysis")

        if generate:
            trait_lookup = load_trait_database(TRAIT_DB_PATH)

            if uploaded and not use_demo:
                temp_path = "uploaded_genome.txt"
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getvalue())
                genotype_path = temp_path
            else:
                genotype_path = "test_genotype.txt"

            try:
                variants = parse_genotype_file(genotype_path)
                matched_traits = match_traits(trait_lookup, variants)
                report = build_report_object(matched_traits)

                # Save for chatbot
                st.session_state.last_report = report

                if not matched_traits:
                    st.warning("No traits were matched. Check that the file uses the expected rsIDs and genotypes.")
                else:
                    with st.spinner("Asking the model to summarize your traits..."):
                        ai_summary = generate_ai_summary(report)

                    st.session_state.last_ai_summary = ai_summary

                    st.markdown("### AI overview")
                    if ai_summary:
                        st.write(ai_summary)
                    else:
                        st.write(
                            "The AI overview could not be generated, but the structured trait report is still available below."
                        )

                    text_report = generate_text_report(report)

                    with st.expander("View technical text summary"):
                        st.text(text_report)

                    st.markdown("### Detailed trait report")
                    html_report = generate_html_report(report, ai_summary=ai_summary)
                    st.components.v1.html(html_report, height=850, scrolling=True)

                    if email_for_result.strip():
                        st.info(
                            f"In a future deployed version, this report could also be sent securely to {email_for_result.strip()} "
                            "from a GenAI Engine email address."
                        )

            except Exception as e:
                st.error(f"Something went wrong while generating the report: {e}")

    with col2:
        st.markdown("#### What your report looks like")
        st.markdown(
            """
            You will receive:
            
            - A plain-language AI overview of your matched traits  
            - Individual trait cards showing rsIDs, genes, and genotype  
            - Short explanations of what each trait may mean in everyday life  
            - A scannable HTML layout that you can revisit later  
            """
        )
        st.markdown("---")
        st.markdown("#### What GenAI Engine does")
        st.markdown(
            """
            - Parses a raw genotype file line by line  
            - Matches known variants against a curated trait database  
            - Builds a structured JSON representation of the trait profile  
            - Uses a language model to write an educational overview  
            - Renders a printable HTML report  
            """
        )
        st.markdown("---")
        st.markdown("#### What GenAI Engine does not do")
        st.markdown(
            """
            - It does not diagnose conditions  
            - It does not replace medical care  
            - It does not cover the entire genome  
            """
        )
        st.caption(
            "Always speak with a licensed clinician or genetic counselor for health decisions."
        )

# ---------- LIFESTYLE CHATBOT ----------
elif page == "Lifestyle Chatbot":
    st.markdown('<div class="section-title">Lifestyle chatbot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Ask questions about your trait report, habits, and what you might want to pay attention to. '
        'The chatbot is designed for education and lifestyle ideas, not for medical advice.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "This chatbot can discuss habits, routines, and ways to think about your traits, but it cannot provide medical advice, "
        "diagnose conditions, or tell you what to do with medications. Always talk with a healthcare professional for medical decisions."
    )

    st.markdown("#### Lifestyle overview plan (beta)")

    # Show previously generated plan if available
    existing_plan = st.session_state.get("lifestyle_plan")
    if existing_plan:
        with st.expander("View your current lifestyle plan", expanded=True):
            st.markdown(existing_plan)

    st.caption(
        "This plan is educational only. It suggests gentle habit ideas based on your traits and should not be treated as medical guidance."
    )

    generate_plan_clicked = st.button("Generate / refresh my lifestyle plan")

    report = st.session_state.get("last_report")
    ai_summary = st.session_state.get("last_ai_summary")

    if report is None:
        st.info("Generate a report on the Upload page first so the chatbot has context about your traits.")
    else:
        # Optionally generate a structured lifestyle overview plan
        if generate_plan_clicked:
            lifestyle_context = f"Trait JSON:\n{report}\n\nAI summary of traits:\n{ai_summary or ''}"
            lifestyle_system = (
                "You are a careful genetics-informed lifestyle coach. "
                "Given a structured trait report and an AI summary, create a short, non-medical lifestyle plan. "
                "Organize the plan into sections such as Sleep, Focus & Learning, Movement & Recovery, Caffeine & Stimulants, "
                "and Everyday Habits. For each section, list 3-5 gentle, practical ideas that could be helpful for someone with these traits. "
                "Use tentative language (may, might, could) and remind the reader that this is not medical advice."
            )

            try:
                plan_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": lifestyle_system},
                        {"role": "user", "content": lifestyle_context},
                    ],
                    temperature=0.6,
                )
                lifestyle_plan = plan_resp.choices[0].message.content.strip()
                st.session_state["lifestyle_plan"] = lifestyle_plan

                with st.expander("View your current lifestyle plan", expanded=True):
                    st.markdown(lifestyle_plan)
            except Exception as e:
                st.error(f"There was an error generating the lifestyle plan: {e}")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show previous messages
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)

        user_input = st.chat_input("Ask a question about your traits or lifestyle")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            system_prompt = (
                "You are a genetics informed lifestyle coach. "
                "You receive a JSON report of traits and a short AI summary. "
                "You may discuss possible lifestyle ideas related to sleep, focus, caffeine, training, and general wellness. "
                "You must avoid medical advice, diagnosis, or treatment recommendations. "
                "Use careful language like may, might, and could, and encourage the user to talk with a clinician "
                "or genetic counselor for any medical questions."
            )

            context_snippet = f"Trait JSON:\n{report}\n\nSummary:\n{ai_summary or ''}"

            with st.chat_message("assistant"):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": context_snippet},
                            {"role": "user", "content": user_input},
                        ],
                        temperature=0.7,
                    )
                    reply = resp.choices[0].message.content.strip()
                except Exception as e:
                    reply = f"There was an error calling the model: {e}"

                st.markdown(reply)
                st.session_state.chat_history.append(("assistant", reply))

 # ---------- TRAIT EXPLORER ----------
elif page == "Trait Explorer":
    st.markdown('<div class="section-title">Trait explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Browse and search the underlying trait database used by the interpreter. '
        'This view is for learning and debugging, not for medical interpretation.</div>',
        unsafe_allow_html=True,
    )

    # Load trait rows
    trait_rows = load_trait_rows_from_csv(TRAIT_DB_PATH)

    if not trait_rows:
        st.info("No trait rows could be loaded from the database.")
    else:
        # Build simple filter options
        all_categories = sorted({r.get("category", "") for r in trait_rows if r.get("category")})
        all_evidence = sorted({r.get("evidence_strength", "") for r in trait_rows if r.get("evidence_strength")})

        col_search, col_cat, col_ev = st.columns([1.4, 0.8, 0.8])
        with col_search:
            query = st.text_input("Search by rsID, gene, or trait name")
        with col_cat:
            cat_filter = st.multiselect("Filter by category", options=all_categories)
        with col_ev:
            ev_filter = st.multiselect("Filter by evidence", options=all_evidence)

        # Apply filters
        def match_row(row):
            text = (row.get("rsid", "") + " " + row.get("gene", "") + " " + row.get("trait_name", "")).lower()
            if query and query.lower() not in text:
                return False
            if cat_filter and row.get("category") not in cat_filter:
                return False
            if ev_filter and row.get("evidence_strength") not in ev_filter:
                return False
            return True

        filtered = [r for r in trait_rows if match_row(r)]

        st.markdown(f"Showing **{len(filtered)}** of **{len(trait_rows)}** traits.")

        if filtered:
            # Show a compact table
            display_cols = [c for c in ["trait_id", "trait_name", "category", "rsid", "gene", "genotype", "effect_label", "evidence_strength"] if c in filtered[0]]
            st.dataframe(filtered, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### JSON model preview for the first filtered trait")

            first = filtered[0]
            # Convert a CSV row into a JSON-like trait object
            json_trait = {
                "rsid": first.get("rsid"),
                "gene": first.get("gene"),
                "trait_category": first.get("category"),
                "trait_name": first.get("trait_name"),
                "genotype": first.get("genotype"),
                "variant_effect": first.get("effect_label"),
                "evidence_level": first.get("evidence_strength"),
                "explanation": first.get("explanation"),
                "mechanism": first.get("mechanism") or "",  # optional column
                "lifestyle_links": [],  # can be populated later
                "notes": "For education only; not a diagnosis.",
            }

            st.code(json.dumps(json_trait, indent=2), language="json")

            # Optional: export full JSON model
            if st.button("Export full trait database as JSON model"):
                json_model = []
                for row in filtered:
                    obj = {
                        "rsid": row.get("rsid"),
                        "gene": row.get("gene"),
                        "trait_category": row.get("category"),
                        "trait_name": row.get("trait_name"),
                        "genotype": row.get("genotype"),
                        "variant_effect": row.get("effect_label"),
                        "evidence_level": row.get("evidence_strength"),
                        "explanation": row.get("explanation"),
                        "mechanism": row.get("mechanism") or "",
                        "lifestyle_links": [],
                        "notes": "For education only; not a diagnosis.",
                    }
                    json_model.append(obj)

                try:
                    with open("trait_database_model.json", "w", encoding="utf-8") as f:
                        json.dump(json_model, f, ensure_ascii=False, indent=2)
                    st.success("Exported JSON model to trait_database_model.json in the project folder.")
                except Exception as e:
                    st.error(f"Could not write JSON model file: {e}")
        else:
            st.info("No traits matched your filters.")


# ---------- TRAIT SCIENCE ----------
elif page == "Trait Science":
    st.markdown('<div class="section-title">Science behind the traits</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A high-level overview of how GenAI Engine turns individual genetic variants into trait interpretations.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### What is a SNP?", unsafe_allow_html=True)
    st.markdown(
        """
        A single nucleotide polymorphism (SNP) is a position in the genome where people commonly differ by a single base.
        GenAI Engine uses SNPs that have been studied in the literature and are associated with traits such as caffeine
        metabolism, sleep patterns, exercise response, and certain neurobehavioral tendencies.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How traits are defined", unsafe_allow_html=True)
    st.markdown(
        """
        Each trait in the internal database links:
        
        - One or more rsIDs (SNP identifiers)  
        - The gene or genomic region  
        - A description of the reported effect (for example, typical vs. increased sensitivity)  
        - A qualitative evidence label that reflects the strength and consistency of published findings  
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What this panel focuses on", unsafe_allow_html=True)
    st.markdown(
        """
        The current prototype focuses on a small set of traits that connect to everyday lifestyle questions, such as:
        
        - Sleep timing and sleep depth tendencies  
        - Response to caffeine and stimulants  
        - Exercise and recovery related traits  
        - Certain sensory and neurobehavior-related features  
        
        These are chosen because they are easier to connect to day-to-day habits and are safer to discuss in an
        educational context than serious disease risk.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Why nothing here is a diagnosis", unsafe_allow_html=True)
    st.markdown(
        """
        Most common genetic variants have small effects that interact with many other factors like sleep, stress,
        environment, and medical history. GenAI Engine treats traits as gentle hints rather than answers. Any serious
        medical concern should always be discussed with a healthcare professional.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Data model sketch", unsafe_allow_html=True)
    st.markdown(
        """
        Internally, each trait can be represented as a small JSON object that connects the genetic signal to a lifestyle-oriented
        explanation. A simplified example structure might look like:
        """,
        unsafe_allow_html=True,
    )

    st.code(
        '{\n'
        '  "rsid": "rs1234",\n'
        '  "gene": "ADORA2A",\n'
        '  "trait_category": "Caffeine sensitivity",\n'
        '  "variant_effect": "Increased sensitivity to caffeine",\n'
        '  "mechanism": "Adenosine receptor signaling differences",\n'
        '  "evidence_level": "Moderate",\n'
        '  "lifestyle_links": [\n'
        '    "May feel stronger effects from caffeine",\n'
        '    "Might benefit from limiting caffeine later in the day"\n'
        '  ],\n'
        '  "notes": "For education only; not a diagnosis."\n'
        '}',
        language="json",
    )

    st.markdown(
        """
        Over time, this kind of structure can be expanded into a richer knowledge graph that links traits to sources, study quality,
        and more nuanced lifestyle ideas, while still keeping the interface simple for end users.
        """,
        unsafe_allow_html=True,
    )

# ---------- ABOUT ----------
elif page == "About":
    st.markdown('<div class="section-title">About GenAI Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'GenAI Engine is an educational prototype built to explore how AI can help interpret '
        'small scale genomic trait panels in a transparent and lifestyle aware way.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Concept", unsafe_allow_html=True)
        st.markdown(
            """
            This project combines three layers:
            1. A structured genetics engine that maps specific SNPs to trait interpretations  
            2. A curated database with gene, rsID, and literature based explanations  
            3. A language model that converts this structure into a narrative overview focused on education and lifestyle  
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Design principles", unsafe_allow_html=True)
        st.markdown(
            """
            - Educational first: explain, do not prescribe  
            - Transparent: the underlying trait data stays visible  
            - Lifestyle aware: DNA is framed as one input among sleep, habits, and environment  
            - Human centered: final decisions belong to people, not models  
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Technical sketch", unsafe_allow_html=True)
    st.code(
        """User file (.txt) 
   ↓
Parse SNPs and genotypes
   ↓
Match against trait_database.csv
   ↓
Build JSON report object
   ↓
Call OpenAI model for overview text
   ↓
Render HTML report (and optional PDF)""",
        language="text",
    )

    # Safety and limitations (About)
    st.markdown('<div class="section-title">Safety and limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        GenAI Engine is designed as an educational tool. It highlights potential trait patterns and how they might relate to
        lifestyle, but it does not measure risk for disease or replace professional genetic counseling. All interpretations
        are simplified and should be viewed as conversation starters, not conclusions.
        """,
        unsafe_allow_html=True,
    )

    # FAQ (About - extended)
    st.markdown('<div class="section-title">Frequently asked questions</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **Does GenAI Engine store my DNA data?**  
        In this prototype, files are handled within your session. A production deployment would include a clear privacy
        policy and options for data deletion or local-only processing.

        **Can this tell me what conditions I have or will develop?**  
        No. The trait panel is limited and focuses on common variants related to lifestyle and tendencies. Medical genetics
        requires much deeper analysis and professional interpretation.

        **Is AI making up results?**  
        The AI model is used to turn structured trait data into readable text. The underlying rsIDs, genes, and effect
        labels come from a curated table, not from the model inventing variants.

        **Who is this for?**  
        Students, early researchers, and curious individuals who want to practice thinking about genomics and lifestyle
        in a careful, low-stakes way.
        """,
        unsafe_allow_html=True,
    )

# ---------- CONTACT ----------
elif page == "Contact":
    st.markdown('<div class="section-title">Contact</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'Interested in the science, the code, or future collaborations around AI, genomics, and lifestyle? '
        'Use the form below to leave a message.</div>',
        unsafe_allow_html=True,
    )

    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        role = st.selectbox(
            "I am primarily a…",
            ["Student", "Educator", "Researcher or clinician", "Developer", "Other"],
        )
        message = st.text_area("Message", height=140)
        submitted = st.form_submit_button("Send message")

        if submitted:
            if not (name.strip() and email.strip() and message.strip()):
                st.warning("Please fill in name, email, and a brief message.")
            else:
                st.success(
                    "Thank you for reaching out. In a production environment this would send your message to the project owner."
              )

st.markdown("</div>", unsafe_allow_html=True)
# ---------- Global footer ----------
st.markdown(
    """
    <div class="site-footer">
      <div>
        GenAI Engine · Educational genomics prototype · Not for diagnosis or treatment.
      </div>
      <div>
        Built by Anjali Nalla · Independent project, not affiliated with 23andMe or any consumer genetics company.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")
st.caption("Prototype only. No data is persisted or transmitted beyond this session.")
import os
import re
import csv
import html
import difflib
import numpy as np
import pandas as pd
import streamlit as st

from google_play_scraper import reviews, Sort

import emoji
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# =========================================================
# 1) PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Sentimen Analyzer", page_icon="💬", layout="wide")


# =========================================================
# 2) SINGLE CSS INJECT (konsisten warna + spacing)
# =========================================================
def inject_css():
    CSS = """
    <style>
    :root{
      --bg: #F8FAFC;
      --card: #FFFFFF;
      --text: #0F172A;
      --muted: #475569;
      --border: #E2E8F0;
      --border2: #E6F0FF;
      --shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
      --radius: 16px;
      --radius-sm: 12px;
      --primary1: #1565C0;
      --primary2: #00BFA5;
      --soft: #F3F8FF;
      --soft2: #F1F5F9;
      --danger-bg: #FEE2E2;
      --danger-br: #FCA5A5;
      --success-bg: #DCFCE7;
      --success-br: #86EFAC;
    }

    /* layout spacing */
    .block-container{
      padding-top: 2.4rem !important;
      padding-bottom: 2rem !important;
    }

    /* Header */
    .title-grad{
      font-size: 34px;
      font-weight: 900;
      margin: 0;
      line-height: 1.1;
      background: linear-gradient(90deg, var(--primary1), var(--primary2));
      -webkit-background-clip:text;
      -webkit-text-fill-color:transparent;
    }
    .subtle{
      color: var(--muted);
      font-size: 14px;
      margin-top: 6px;
    }
    .divider{
      height: 1px;
      background: var(--border2);
      margin: 12px 0 18px 0;
    }

    /* Cards */
    .card{
      background: var(--card);
      border-radius: var(--radius);
      padding: 18px;
      border: 1px solid var(--border2);
      box-shadow: var(--shadow);
    }
    .card-tight{
      padding: 14px 16px;
    }

    /* Buttons (global) */
    .stButton button{
      background: linear-gradient(90deg, var(--primary1), var(--primary2)) !important;
      color: white !important;
      border: 0 !important;
      border-radius: var(--radius-sm) !important;
      padding: 0.75rem 1rem !important;
      font-weight: 800 !important;
    }

    /* Segmented control (radio horizontal) */
    div[data-testid="stRadio"] > div{
      background: var(--card);
      border: 1px solid var(--border2);
      border-radius: var(--radius);
      padding: 10px 12px;
      box-shadow: var(--shadow);
    }
    div[data-testid="stRadio"] label{
      font-weight: 800 !important;
      color: var(--text) !important;
    }

    /* Stepper */
    .stepper{
      display:flex; gap:10px; flex-wrap:wrap;
      background: var(--card);
      border: 1px solid var(--border2);
      border-radius: var(--radius);
      padding: 12px;
      box-shadow: var(--shadow);
      align-items:center;
    }
    .step{
      display:flex; align-items:center; gap:10px;
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--soft2);
      color: var(--text);
      font-weight: 900;
      font-size: 13px;
    }
    .step .dot{
      width: 10px; height: 10px; border-radius: 999px;
      background: var(--border);
    }
    .step.active{
      border: 0;
      background: linear-gradient(90deg, var(--primary1), var(--primary2));
      color: white;
    }
    .step.active .dot{ background: rgba(255,255,255,0.9); }
    .step.done{
      background: var(--success-bg);
      border: 1px solid var(--success-br);
    }
    .step.done .dot{ background: #16A34A; }

    /* Metrics cards */
    .metric-card{
      background: var(--card);
      border: 1px solid var(--border2);
      border-radius: var(--radius);
      padding: 14px 16px;
      box-shadow: var(--shadow);
    }
    .metric-title{font-weight: 900; color: var(--text); margin: 0; font-size: 14px;}
    .metric-value{font-weight: 900; font-size: 26px; margin: 4px 0 0 0;}
    .metric-sub{color: var(--muted); margin-top: 6px; font-size: 13px;}

    /* Diff highlight */
    .diff-box{
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 12px;
    }
    .diff-before, .diff-after{
      font-size: 14px; line-height: 1.5; margin: 6px 0;
    }
    .diff-tag{
      font-weight: 900; color: var(--text); margin-right: 8px;
    }
    .diff-add{
      background: var(--success-bg);
      border: 1px solid var(--success-br);
      padding: 0 4px;
      border-radius: 6px;
    }
    .diff-del{
      background: var(--danger-bg);
      border: 1px solid var(--danger-br);
      padding: 0 4px;
      border-radius: 6px;
      text-decoration: line-through;
    }

    /* Sidebar compact */
    section[data-testid="stSidebar"] .block-container{
      padding-top: 0.8rem;
      padding-bottom: 0.8rem;
    }
    section[data-testid="stSidebar"] hr{
      margin: 0.6rem 0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button{
      padding: 0.55rem 0.7rem !important;
      font-size: 0.92rem !important;
      border-radius: var(--radius-sm) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label{
      font-size: 0.92rem !important;
    }
    section[data-testid="stSidebar"] .stCaption{
      font-size: 0.8rem !important;
    }
    section[data-testid="stSidebar"] div[role="progressbar"]{
      height: 8px !important;
    }
    </style>
    """
    st.markdown(CSS, unsafe_allow_html=True)


inject_css()


# =========================================================
# 3) UI helpers
# =========================================================
def card_open(tight=False):
    cls = "card card-tight" if tight else "card"
    st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def bright_header(title: str, subtitle: str):
    st.markdown(f"<p class='title-grad'>{title}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtle'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

def render_stepper(current_index: int, done_mask: list[bool]):
    labels = ["Home", "Dataset", "Preprocessing", "SVM"]
    dots = []
    for i, lab in enumerate(labels):
        cls = "step"
        if done_mask[i]:
            cls += " done"
        if i == current_index:
            cls += " active"
        dots.append(f"<div class='{cls}'><span class='dot'></span>{lab}</div>")
    st.markdown(f"<div class='stepper'>{''.join(dots)}</div>", unsafe_allow_html=True)


# =========================================================
# 4) Session state
# =========================================================
def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "Home"  # wizard target
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "text_col" not in st.session_state:
        st.session_state.text_col = None
    if "prep_steps" not in st.session_state:
        st.session_state.prep_steps = {}
    if "final_df" not in st.session_state:
        st.session_state.final_df = None
    if "eval_df" not in st.session_state:
        st.session_state.eval_df = None  # hasil klasifikasi (test set + pred)
    if "svm_model_info" not in st.session_state:
        st.session_state.svm_model_info = None  # simpan metrics ringkas

init_state()


# =========================================================
# 5) Utilities
# =========================================================
def ensure_nltk():
    try:
        _ = stopwords.words("indonesian")
    except LookupError:
        nltk.download("stopwords")

def to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)

@st.cache_resource
def get_stemmer():
    return StemmerFactory().create_stemmer()


# =========================================================
# 6) Scraping helper (cache)
# =========================================================
@st.cache_data(show_spinner=False)
def scrape_google_play(app_id: str, jumlah: int, bahasa="id", negara="id"):
    all_rows = []
    next_token = None

    while len(all_rows) < jumlah:
        batch, next_token = reviews(
            app_id,
            lang=bahasa,
            country=negara,
            sort=Sort.NEWEST,
            count=min(200, jumlah - len(all_rows)),
            continuation_token=next_token,
        )
        if not batch:
            break

        for r in batch:
            all_rows.append(
                {
                    "userName": r.get("userName"),
                    "score": r.get("score"),
                    "at": r.get("at"),
                    "content": r.get("content"),
                }
            )

        if next_token is None:
            break

    return pd.DataFrame(all_rows)


# =========================================================
# 7) Preprocessing pipeline (aman)
# =========================================================
def case_folding(text: str) -> str:
    return to_text(text).lower()

def load_kamus_excel_safe(path: str) -> dict:
    try:
        df = pd.read_excel(path)
        if "non_standard" in df.columns and "standard_word" in df.columns:
            df = df[["non_standard", "standard_word"]].dropna()
            return dict(zip(df["non_standard"].astype(str), df["standard_word"].astype(str)))
        if df.shape[1] >= 2:
            df2 = df.iloc[:, :2].dropna()
            return dict(zip(df2.iloc[:, 0].astype(str), df2.iloc[:, 1].astype(str)))
        return {}
    except Exception:
        return {}

def normalisasi_kamus(text: str, kamus: dict) -> str:
    text = to_text(text)
    if not kamus:
        return text
    words = text.split()
    return " ".join([kamus.get(w, w) for w in words])

def data_cleansing(text: str) -> str:
    text = to_text(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stopword_removal(text: str) -> str:
    sw = set(stopwords.words("indonesian"))
    tokens = to_text(text).split()
    tokens = [t for t in tokens if t not in sw]
    return " ".join(tokens)

def stemming(text: str) -> str:
    return get_stemmer().stem(to_text(text))

def tokenizing(text: str):
    t = to_text(text).strip()
    return t.split() if t else []


# =========================================================
# 8) Labeling (lexicon, aman + fallback)
# =========================================================
def load_lexicon_safe(path: str) -> dict:
    lex = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                w = str(row[0]).strip()
                s = str(row[1]).strip()
                if not w:
                    continue
                try:
                    lex[w] = int(float(s))
                except:
                    pass
    except Exception:
        pass
    return lex

def label_by_lexicon(tokens, lex_pos: dict, lex_neg: dict):
    score = 0
    for t in tokens:
        if t in lex_pos:
            score += lex_pos[t]
        if t in lex_neg:
            score += lex_neg[t]

    if score > 0:
        lab = "positif"
    elif score < 0:
        lab = "negatif"
    else:
        lab = "netral"
    return score, lab


# =========================================================
# 9) Diff helpers (sudah ada, dipakai ulang)
# =========================================================
def compute_changes(before_series: pd.Series, after_series: pd.Series, max_examples: int = 6):
    b = before_series.astype(str).fillna("")
    a = after_series.astype(str).fillna("")
    changed_mask = b.ne(a)
    changed_count = int(changed_mask.sum())
    total = int(len(b))
    examples = pd.DataFrame({
        "Sebelum": b[changed_mask].head(max_examples).values,
        "Sesudah": a[changed_mask].head(max_examples).values,
    })
    return changed_count, total, examples

def diff_words_html(before_text: str, after_text: str) -> tuple[str, str]:
    b_tokens = to_text(before_text).split()
    a_tokens = to_text(after_text).split()

    sm = difflib.SequenceMatcher(a=b_tokens, b=a_tokens)
    b_out, a_out = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        b_chunk = [html.escape(t) for t in b_tokens[i1:i2]]
        a_chunk = [html.escape(t) for t in a_tokens[j1:j2]]

        if tag == "equal":
            b_out.extend(b_chunk)
            a_out.extend(a_chunk)
        elif tag == "delete":
            b_out.extend([f"<span class='diff-del'>{t}</span>" for t in b_chunk])
        elif tag == "insert":
            a_out.extend([f"<span class='diff-add'>{t}</span>" for t in a_chunk])
        elif tag == "replace":
            b_out.extend([f"<span class='diff-del'>{t}</span>" for t in b_chunk])
            a_out.extend([f"<span class='diff-add'>{t}</span>" for t in a_chunk])

    return " ".join(b_out), " ".join(a_out)

def show_change_summary_and_examples(before_df: pd.DataFrame, after_df: pd.DataFrame, col: str = "content"):
    changed_count, total, examples = compute_changes(before_df[col], after_df[col], max_examples=6)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total baris", total)
    c2.metric("Baris berubah", changed_count)
    pct = (changed_count / total * 100) if total else 0
    c3.metric("Persentase berubah", f"{pct:.2f}%")

    if changed_count == 0:
        st.info("Tidak ada perubahan pada tahap ini.")
        return

    st.markdown("**Contoh perubahan (kata ditambah hijau, kata dihapus merah):**")
    for _, row in examples.iterrows():
        b_html, a_html = diff_words_html(row["Sebelum"], row["Sesudah"])
        st.markdown(
            f"""
            <div class="diff-box">
              <div class="diff-before"><span class="diff-tag">Sebelum:</span>{b_html}</div>
              <div class="diff-after"><span class="diff-tag">Sesudah:</span>{a_html}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 10) NEW: Preprocessing "impact" ringkas per step
# =========================================================
def text_stats(series: pd.Series):
    s = series.astype(str).fillna("")
    lengths = s.str.len()
    token_counts = s.apply(lambda x: len(x.split()))
    joined = " ".join(s.tolist()).split()
    uniq = len(set(joined)) if joined else 0
    return {
        "avg_chars": float(lengths.mean()) if len(lengths) else 0.0,
        "avg_tokens": float(token_counts.mean()) if len(token_counts) else 0.0,
        "unique_words": int(uniq),
        "empty_rows": int((s.str.strip() == "").sum()),
    }

def top_token_delta(before_series: pd.Series, after_series: pd.Series, top_k=10):
    # token frequency diff: after - before
    def freq(s):
        toks = " ".join(s.astype(str).fillna("").tolist()).split()
        return pd.Series(toks).value_counts() if toks else pd.Series(dtype=int)

    fb = freq(before_series)
    fa = freq(after_series)
    delta = fa.sub(fb, fill_value=0).sort_values(ascending=False)
    added = delta[delta > 0].head(top_k)
    removed = delta[delta < 0].head(top_k).abs()
    return added, removed

def show_impact(before_df: pd.DataFrame, after_df: pd.DataFrame, step_name: str, col="content"):
    b = text_stats(before_df[col])
    a = text_stats(after_df[col])

    st.markdown("#### Ringkasan impact (sebelum → sesudah)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rata-rata karakter", f"{b['avg_chars']:.1f}", f"{a['avg_chars']-b['avg_chars']:+.1f}")
    c2.metric("Rata-rata token", f"{b['avg_tokens']:.1f}", f"{a['avg_tokens']-b['avg_tokens']:+.1f}")
    c3.metric("Kata unik", f"{b['unique_words']}", f"{a['unique_words']-b['unique_words']:+d}")
    c4.metric("Baris kosong", f"{b['empty_rows']}", f"{a['empty_rows']-b['empty_rows']:+d}")

    # untuk step yang memang mengubah token (cleansing/stopword/stemming/normalisasi)
    if any(k in step_name.lower() for k in ["cleansing", "stopword", "stemming", "normalisasi", "case folding"]):
        added, removed = top_token_delta(before_df[col], after_df[col], top_k=10)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Top kata bertambah**")
            if len(added):
                st.dataframe(added.rename("Δ Frek").reset_index().rename(columns={"index": "Kata"}), use_container_width=True)
            else:
                st.caption("Tidak ada perubahan signifikan.")
        with cc2:
            st.markdown("**Top kata berkurang/hilang**")
            if len(removed):
                st.dataframe(removed.rename("Δ Frek").reset_index().rename(columns={"index": "Kata"}), use_container_width=True)
            else:
                st.caption("Tidak ada perubahan signifikan.")


# =========================================================
# 11) Sidebar: progress + reset
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Kontrol")
    done = [False, False, False, False]
    done[1] = st.session_state.raw_df is not None
    done[2] = st.session_state.final_df is not None
    done[3] = st.session_state.eval_df is not None

    step_done = sum(done[1:])  # dataset/prep/svm
    st.markdown("**📈 Progress**")
    st.progress(step_done / 3)
    st.caption(f"{step_done}/3 (Dataset → Preprocessing → SVM)")

    if st.button("🔄 Reset semua", use_container_width=True):
        st.session_state.raw_df = None
        st.session_state.text_col = None
        st.session_state.prep_steps = {}
        st.session_state.final_df = None
        st.session_state.eval_df = None
        st.session_state.svm_model_info = None
        st.session_state.page = "Home"
        st.rerun()


# =========================================================
# 12) Navigation: Segmented control modern + wizard Next/Back
# =========================================================
pages = ["Home", "Dataset", "Preprocessing", "SVM"]

# Segmented control (radio horizontal) — sinkron dengan wizard
selected = st.radio(
    "Navigasi",
    pages,
    horizontal=True,
    index=pages.index(st.session_state.page),
    label_visibility="collapsed",
)
if selected != st.session_state.page:
    st.session_state.page = selected
    st.rerun()

# Stepper visual
current_idx = pages.index(st.session_state.page)
done_mask = [False, st.session_state.raw_df is not None, st.session_state.final_df is not None, st.session_state.eval_df is not None]
render_stepper(current_idx, done_mask)
st.markdown("")


def wizard_nav(next_allowed=True, prev_allowed=True):
    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("⬅️ Back", use_container_width=True, disabled=not prev_allowed):
            st.session_state.page = pages[max(0, current_idx - 1)]
            st.rerun()
    with c2:
        if st.button("Next ➡️", use_container_width=True, disabled=not next_allowed):
            st.session_state.page = pages[min(len(pages) - 1, current_idx + 1)]
            st.rerun()


# =========================================================
# 13) HOME
# =========================================================
if st.session_state.page == "Home":
    bright_header("💬 Sentimen Analyzer", "Ambil dataset → preprocessing bertahap → klasifikasi SVM (hasil langsung).")

    card_open()
    st.markdown(
        """
#### Yang bisa kamu lakukan
- **Dataset**: scraping ulasan Google Play atau upload CSV/Excel.
- **Preprocessing**: tampil **bertahap**, ada ringkasan impact + contoh perubahan.
- **Klasifikasi SVM**: hasil mudah dipahami + bisa download CSV.
        """.strip()
    )
    card_close()

    st.markdown("")
    wizard_nav(next_allowed=True, prev_allowed=False)


# =========================================================
# 14) DATASET
# =========================================================
elif st.session_state.page == "Dataset":
    bright_header("📦 Dataset", "Pilih sumber dataset, lalu pilih kolom teks untuk dianalisis.")

    tab1, tab2 = st.tabs(["🕷️ Scraping Google Play", "📤 Upload CSV/Excel"])

    with tab1:
        card_open()
        st.markdown("#### Scraping Ulasan Google Play")
        app_id = st.text_input(
            "App ID Google Play",
            value="co.id.bankbsi.superapp",
            help="App ID default sudah diisi untuk Bank BSI Super App"
        )
        jumlah = st.number_input("Jumlah ulasan", min_value=50, max_value=5000, value=200, step=50)
        bahasa = st.selectbox("Bahasa", ["id", "en"], index=0)
        negara = st.selectbox("Negara", ["id", "us", "sg", "my"], index=0)

        c1, c2 = st.columns(2)
        do_scrape = c1.button("🧲 Mulai Scraping", use_container_width=True)
        do_reset = c2.button("🧹 Reset Dataset", use_container_width=True)

        if do_reset:
            st.session_state.raw_df = None
            st.session_state.text_col = None
            st.session_state.prep_steps = {}
            st.session_state.final_df = None
            st.session_state.eval_df = None
            st.session_state.svm_model_info = None
            st.rerun()

        if do_scrape:
            if not app_id.strip():
                st.error("App ID tidak boleh kosong.")
            else:
                with st.spinner("Mengambil ulasan..."):
                    df = scrape_google_play(app_id.strip(), int(jumlah), bahasa=bahasa, negara=negara)
                if df.empty:
                    st.warning("Data kosong. Coba App ID lain.")
                else:
                    st.session_state.raw_df = df
                    st.session_state.text_col = "content" if "content" in df.columns else df.columns[0]
                    st.session_state.prep_steps = {}
                    st.session_state.final_df = None
                    st.session_state.eval_df = None
                    st.session_state.svm_model_info = None
                    st.success(f"Berhasil mengambil {len(df)} ulasan.")
        card_close()

    with tab2:
        card_open()
        file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
        if file is not None:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                st.session_state.raw_df = df
                cols = list(df.columns)
                st.session_state.text_col = "content" if "content" in cols else cols[0]
                st.session_state.prep_steps = {}
                st.session_state.final_df = None
                st.session_state.eval_df = None
                st.session_state.svm_model_info = None
                st.success(f"Dataset berhasil di-load: {len(df)} baris.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
        card_close()

    if st.session_state.raw_df is not None:
        st.markdown("")
        card_open()
        st.markdown("#### Preview dataset")
        st.dataframe(st.session_state.raw_df.head(50), use_container_width=True)
        card_close()

        st.markdown("")
        card_open()
        st.markdown("#### Pilih kolom teks untuk analisis")
        cols = list(st.session_state.raw_df.columns)
        idx = cols.index(st.session_state.text_col) if st.session_state.text_col in cols else 0
        st.session_state.text_col = st.selectbox("Kolom teks", cols, index=idx)
        card_close()

    st.markdown("")
    next_ok = (st.session_state.raw_df is not None and st.session_state.text_col is not None)
    wizard_nav(next_allowed=next_ok, prev_allowed=True)


# =========================================================
# 15) PREPROCESSING
# =========================================================
elif st.session_state.page == "Preprocessing":
    bright_header("🧼 Preprocessing", "Klik tombol untuk memproses. Tiap langkah tampil impact + contoh perubahan.")

    if st.session_state.raw_df is None or not st.session_state.text_col:
        st.warning("Dataset/kolom teks belum siap. Kembali ke menu Dataset.")
        wizard_nav(next_allowed=False, prev_allowed=True)
        st.stop()

    ensure_nltk()

    ASSETS_DIR = "assets"
    KAMUS_PATH = os.path.join(ASSETS_DIR, "kamus.xlsx")
    LEX_POS_PATH = os.path.join(ASSETS_DIR, "positive.csv")
    LEX_NEG_PATH = os.path.join(ASSETS_DIR, "negative.csv")

    ok_kamus = os.path.exists(KAMUS_PATH)
    ok_pos = os.path.exists(LEX_POS_PATH)
    ok_neg = os.path.exists(LEX_NEG_PATH)

    card_open(tight=True)
    st.markdown("#### File pendukung (otomatis)")
    st.write("📘 Kamus:", "✅ ditemukan" if ok_kamus else "⚠️ tidak ada (normalisasi dilewati)")
    st.write("🟢 Lexicon +:", "✅ ditemukan" if ok_pos else "⚠️ tidak ada (label fallback)")
    st.write("🔴 Lexicon -:", "✅ ditemukan" if ok_neg else "⚠️ tidak ada (label fallback)")
    card_close()

    st.markdown("")
    drop_neutral = st.checkbox("Hapus data netral (score=0)", value=True)
    run_prep = st.button("⚙️ Jalankan Preprocessing", use_container_width=True)

    def show_compare(step_title, before_df, after_df, n=15):
        st.markdown("")
        card_open()
        st.markdown(f"### {step_title}")

        # Impact summary
        show_impact(before_df, after_df, step_title, col="content")

        st.markdown("---")
        # Change summary + examples (diff)
        show_change_summary_and_examples(before_df, after_df, col="content")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sebelum**")
            st.dataframe(before_df[["content"]].head(n), use_container_width=True)
        with c2:
            st.markdown("**Sesudah**")
            st.dataframe(after_df[["content"]].head(n), use_container_width=True)
        card_close()

    if run_prep:
        base = st.session_state.raw_df.copy()
        text_col = st.session_state.text_col

        df0 = base.copy()
        df0["content"] = df0[text_col].apply(to_text)

        st.session_state.prep_steps = {}
        st.session_state.final_df = None
        st.session_state.eval_df = None
        st.session_state.svm_model_info = None
        st.session_state.prep_steps["0) Data Awal"] = df0.copy()

        df1 = df0.copy()
        df1["content"] = df1["content"].apply(case_folding)
        st.session_state.prep_steps["1) Case Folding"] = df1.copy()

        df2 = df1.copy()
        df2["content"] = df2["content"].apply(data_cleansing)
        st.session_state.prep_steps["2) Data Cleansing"] = df2.copy()

        kamus = load_kamus_excel_safe(KAMUS_PATH) if ok_kamus else {}
        df3 = df2.copy()
        df3["content"] = df3["content"].apply(lambda x: normalisasi_kamus(x, kamus))
        st.session_state.prep_steps["3) Normalisasi"] = df3.copy()

        df4 = df3.copy()
        df4["content"] = df4["content"].apply(stopword_removal)
        st.session_state.prep_steps["4) Stopword Removal"] = df4.copy()

        df5 = df4.copy()
        df5["content"] = df5["content"].apply(stemming)
        st.session_state.prep_steps["5) Stemming"] = df5.copy()

        df6 = df5.copy()
        df6["tokens"] = df6["content"].apply(tokenizing)
        st.session_state.prep_steps["6) Tokenizing"] = df6.copy()

        lex_pos = load_lexicon_safe(LEX_POS_PATH) if ok_pos else {}
        lex_neg = load_lexicon_safe(LEX_NEG_PATH) if ok_neg else {}

        df7 = df6.copy()
        if lex_pos or lex_neg:
            res = df7["tokens"].apply(lambda t: label_by_lexicon(t, lex_pos, lex_neg))
            df7["score"] = res.apply(lambda x: x[0])
            df7["Sentimen"] = res.apply(lambda x: x[1])
        else:
            df7["score"] = df7["tokens"].apply(lambda t: 1 if len(t) >= 5 else (-1 if 0 < len(t) < 3 else 0))
            df7["Sentimen"] = df7["score"].apply(lambda s: "positif" if s > 0 else ("negatif" if s < 0 else "netral"))

        if drop_neutral:
            df7 = df7[df7["Sentimen"] != "netral"].reset_index(drop=True)

        st.session_state.prep_steps["7) Pelabelan"] = df7.copy()
        st.session_state.final_df = df7.copy()

        st.success("Preprocessing selesai! (lihat ringkasan & perbandingan di bawah)")

    # Tampilkan hasil bertahap
    steps = st.session_state.prep_steps
    if steps:
        # download preprocessing hasil akhir
        st.markdown("")
        card_open(tight=True)
        st.markdown("#### ⬇️ Download hasil preprocessing")
        st.download_button(
            "Download preprocessing_final.csv",
            data=st.session_state.final_df.to_csv(index=False).encode("utf-8"),
            file_name="preprocessing_final.csv",
            mime="text/csv",
            use_container_width=True
        )
        card_close()

        keys = list(steps.keys())
        for i in range(1, len(keys)):
            before = steps[keys[i - 1]]
            after = steps[keys[i]]

            if keys[i].startswith("6) Tokenizing"):
                st.markdown("")
                card_open()
                st.markdown("### 6) Tokenizing")

                # Impact tokenizing: pakai content stats juga, plus contoh token
                show_impact(before, after.assign(content=after["content"]), "6) Tokenizing", col="content")

                st.markdown("**Contoh hasil token (teks → tokens):**")
                st.dataframe(after[["content", "tokens"]].head(12), use_container_width=True)
                card_close()

            elif keys[i].startswith("7) Pelabelan"):
                st.markdown("")
                card_open()
                st.markdown("### 7) Pelabelan Sentimen")

                st.markdown("**Distribusi label:**")
                dist = after["Sentimen"].value_counts().rename_axis("Label").reset_index(name="Jumlah")
                st.dataframe(dist, use_container_width=True)

                st.markdown("---")
                st.markdown("**Contoh hasil pelabelan:**")
                st.dataframe(after[["content", "tokens", "score", "Sentimen"]].head(25), use_container_width=True)
                card_close()
            else:
                show_compare(keys[i], before, after)

    st.markdown("")
    next_ok = st.session_state.final_df is not None
    wizard_nav(next_allowed=next_ok, prev_allowed=True)


# =========================================================
# 16) SVM
# =========================================================
elif st.session_state.page == "SVM":
    bright_header("🧠 Klasifikasi SVM", "Hasil dibuat mudah dipahami + bisa download CSV hasil klasifikasi.")

    if st.session_state.final_df is None:
        st.warning("Data belum siap. Jalankan preprocessing dulu.")
        wizard_nav(next_allowed=False, prev_allowed=True)
        st.stop()

    df = st.session_state.final_df.copy()
    if "content" not in df.columns or "Sentimen" not in df.columns:
        st.error("Kolom wajib tidak ada: butuh 'content' dan 'Sentimen'")
        wizard_nav(next_allowed=False, prev_allowed=True)
        st.stop()

    df = df[df["Sentimen"].isin(["positif", "negatif"])].copy()
    df["content"] = df["content"].astype(str).fillna("")

    card_open()
    st.markdown("#### Jalankan SVM (TF-IDF, split 80/20)")
    run_svm = st.button("🚀 Mulai Klasifikasi SVM", use_container_width=True)
    card_close()

    if run_svm:
        X = df["content"]
        y = df["Sentimen"].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        model = LinearSVC()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        labels = ["negatif", "positif"]
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        total = cm.sum()
        benar = int(cm[0, 0] + cm[1, 1])
        salah = int(cm[0, 1] + cm[1, 0])

        salah_neg_jadi_pos = int(cm[0, 1])
        salah_pos_jadi_neg = int(cm[1, 0])

        if salah_neg_jadi_pos > salah_pos_jadi_neg:
            kesimpulan2 = f"Kesalahan paling sering: **negatif dikira positif** ({salah_neg_jadi_pos} kasus)."
        elif salah_pos_jadi_neg > salah_neg_jadi_pos:
            kesimpulan2 = f"Kesalahan paling sering: **positif dikira negatif** ({salah_pos_jadi_neg} kasus)."
        else:
            kesimpulan2 = "Kesalahan negatif→positif dan positif→negatif jumlahnya mirip."

        # simpan eval_df untuk download
        eval_df = pd.DataFrame({
            "Ulasan": X_test.values,
            "Label Asli": y_test.values,
            "Prediksi Model": y_pred
        })

        # confidence (opsional)
        try:
            scores = model.decision_function(X_test_vec)
            eval_df["Skor Keyakinan"] = np.abs(scores)
        except Exception:
            eval_df["Skor Keyakinan"] = np.nan

        st.session_state.eval_df = eval_df
        st.session_state.svm_model_info = {"accuracy": float(acc), "cm": cm}

    if st.session_state.eval_df is None:
        st.info("Klik tombol untuk menjalankan model.")
        wizard_nav(next_allowed=False, prev_allowed=True)
        st.stop()

    # ===== tampil hasil dari state =====
    eval_df = st.session_state.eval_df
    acc = st.session_state.svm_model_info["accuracy"]
    cm = st.session_state.svm_model_info["cm"]

    st.markdown("")
    card_open()
    st.markdown("### ✅ Ringkasan Hasil")
    st.markdown(f"Model benar menebak **{int(cm[0,0]+cm[1,1])} dari {int(cm.sum())}** ulasan (≈ **{acc*100:.1f}%**).")
    st.markdown(
        "- **Benar** artinya prediksi sama dengan label asli.\n"
        "- **Salah** artinya prediksi berbeda dari label asli."
    )
    card_close()

    st.markdown("")
    card_open()
    st.markdown("### ⬇️ Download hasil klasifikasi")
    st.download_button(
        "Download hasil_klasifikasi_svm.csv",
        data=eval_df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_klasifikasi_svm.csv",
        mime="text/csv",
        use_container_width=True
    )
    card_close()

    # ===== confusion matrix versi kartu =====
    st.markdown("")
    card_open()
    st.markdown("### 🔎 Confusion Matrix (versi mudah)")
    a, b, c, d = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
              <p class="metric-title">✅ Negatif terdeteksi benar</p>
              <p class="metric-value">{a}</p>
              <p class="metric-sub">Ulasan negatif, diprediksi negatif (benar)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:12px;">
              <p class="metric-title">❌ Negatif salah jadi positif</p>
              <p class="metric-value">{b}</p>
              <p class="metric-sub">Ulasan negatif, diprediksi positif (salah)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
              <p class="metric-title">❌ Positif salah jadi negatif</p>
              <p class="metric-value">{c}</p>
              <p class="metric-sub">Ulasan positif, diprediksi negatif (salah)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:12px;">
              <p class="metric-title">✅ Positif terdeteksi benar</p>
              <p class="metric-value">{d}</p>
              <p class="metric-sub">Ulasan positif, diprediksi positif (benar)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.caption("Catatan: 4 kartu di atas adalah bentuk confusion matrix yang dibuat lebih mudah dibaca.")
    card_close()

    # ===== classification report =====
    st.markdown("")
    card_open()
    st.markdown("### 📘 Precision / Recall / F1 (ringkas)")
    with st.expander("Penjelasan sederhana"):
        st.markdown(
            """
- **Precision (ketepatan)**: Kalau model bilang “positif”, seberapa sering itu benar?
- **Recall (kelengkapan)**: Dari semua yang benar-benar “positif”, seberapa banyak yang tertangkap model?
- **F1-Score**: Ringkasan yang menyeimbangkan precision dan recall.
- **Support**: Jumlah data pada kelas tersebut.
            """.strip()
        )
    # hitung ulang report dari eval_df (tanpa perlu simpan model)
    rep = classification_report(eval_df["Label Asli"], eval_df["Prediksi Model"], output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose()
    st.dataframe(rep_df, use_container_width=True)
    card_close()

    # ===== contoh salah prediksi =====
    st.markdown("")
    card_open()
    st.markdown("### ❗ Contoh Ulasan yang Salah Prediksi")
    wrong_df = eval_df[eval_df["Label Asli"] != eval_df["Prediksi Model"]].copy()
    if wrong_df.empty:
        st.success("Tidak ada salah prediksi pada data uji.")
    else:
        st.caption("Beberapa contoh yang membuat model keliru (membantu user awam melihat batasan model).")
        st.dataframe(wrong_df.head(15), use_container_width=True)
    card_close()

    # ===== contoh paling yakin =====
    st.markdown("")
    card_open()
    st.markdown("### ⭐ Prediksi Paling Yakin")
    if "Skor Keyakinan" in eval_df.columns and eval_df["Skor Keyakinan"].notna().any():
        top_conf = eval_df.sort_values("Skor Keyakinan", ascending=False).head(10)
        st.caption("Semakin besar skor keyakinan, semakin yakin model.")
        st.dataframe(top_conf, use_container_width=True)
    else:
        st.info("Skor keyakinan tidak tersedia pada konfigurasi ini.")
    card_close()

    st.markdown("")
    wizard_nav(next_allowed=False, prev_allowed=True)

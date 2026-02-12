import os
import re
import csv
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

import matplotlib.pyplot as plt


# =========================
# Page config
# =========================
st.set_page_config(page_title="Sentimen Analyzer", page_icon="💬", layout="wide")


# =========================
# CSS (kontras aman + rapi)
# =========================
CSS = """
<style>
.block-container{
  padding-top: 3.2rem !important;
  padding-bottom: 2rem !important;
}

.bright-card{
  background:#FFFFFF; border-radius:16px; padding:18px;
  border:1px solid #E6F0FF; box-shadow:0 8px 20px rgba(15, 23, 42, 0.06);
}

.title-grad{
  font-size:34px; font-weight:900; margin:0; line-height:1.1;
  background:linear-gradient(90deg,#1565C0,#00BFA5);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}

.subtle{color:#475569; font-size:14px; margin-top:6px;}
.divider{height:1px; background:#E6F0FF; margin:12px 0 18px 0;}

.stButton button {
  background: linear-gradient(90deg,#1565C0,#00BFA5) !important;
  color: white !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 0.75rem 1rem !important;
  font-weight: 800 !important;
}


/* ===== Sidebar nav modern ===== */
.nav-wrap {
  background: #FFFFFF;
  border: 1px solid #E6F0FF;
  border-radius: 16px;
  padding: 12px;
  box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
}

/* tombol nav (default) */
.nav-wrap .stButton>button {
  width: 100% !important;
  background: #F3F8FF !important;
  color: #0F172A !important;
  border: 1px solid #E6F0FF !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  font-weight: 800 !important;
  box-shadow: none !important;
  margin: 6px 0 !important;
  transition: all 0.15s ease-in-out;
}

.nav-wrap .stButton>button:hover {
  transform: translateY(-1px);
  border-color: #BBD7FF !important;
}

/* tombol nav aktif */
.nav-wrap .nav-active .stButton>button {
  background: linear-gradient(90deg,#1565C0,#00BFA5) !important;
  color: #FFFFFF !important;
  border: 0 !important;
}

/* badge kecil di kanan */
.nav-badge {
  float: right;
  font-size: 12px;
  font-weight: 800;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.25);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<style>
/* ===== Compact Sidebar ===== */
section[data-testid="stSidebar"] .block-container {
  padding-top: 0.8rem;
  padding-bottom: 0.8rem;
}

/* perkecil jarak antar elemen */
section[data-testid="stSidebar"] hr {
  margin: 0.6rem 0 !important;
}

/* tombol sidebar lebih kecil */
section[data-testid="stSidebar"] .stButton > button {
  padding: 0.55rem 0.7rem !important;
  font-size: 0.92rem !important;
  border-radius: 12px !important;
}

/* teks di sidebar sedikit diperkecil */
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
  font-size: 0.92rem !important;
}

/* caption diperkecil */
section[data-testid="stSidebar"] .stCaption {
  font-size: 0.8rem !important;
}

/* progress bar lebih tipis */
section[data-testid="stSidebar"] div[role="progressbar"] {
  height: 8px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ===== Styling Tabs Modern ===== */

/* Container tab */
div[data-testid="stTabs"] {
    margin-top: 10px;
}

/* Tab button */
button[data-baseweb="tab"] {
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 10px 18px !important;
    border-radius: 12px 12px 0 0 !important;
    background-color: #F1F5F9 !important;
    color: #334155 !important;
    border: none !important;
    margin-right: 6px !important;
}

/* Tab aktif */
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg,#1565C0,#00BFA5) !important;
    color: white !important;
}

/* Hilangkan garis bawah default */
div[data-testid="stTabs"] > div > div {
    border-bottom: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.diff-box{
  background:#F8FAFC;
  border:1px solid #E2E8F0;
  border-radius:12px;
  padding:12px;
}
.diff-before, .diff-after{
  font-size:14px;
  line-height:1.5;
  margin: 6px 0;
}
.diff-tag{
  font-weight:800;
  color:#0F172A;
  margin-right:8px;
}
.diff-add{
  background: #DCFCE7;
  border: 1px solid #86EFAC;
  padding: 0px 4px;
  border-radius: 6px;
}
.diff-del{
  background: #FEE2E2;
  border: 1px solid #FCA5A5;
  padding: 0px 4px;
  border-radius: 6px;
  text-decoration: line-through;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.metric-card{
  background:#FFFFFF;
  border:1px solid #E6F0FF;
  border-radius:16px;
  padding:14px 16px;
  box-shadow:0 8px 20px rgba(15, 23, 42, 0.06);
}
.metric-title{font-weight:900; color:#0F172A; margin:0; font-size:14px;}
.metric-value{font-weight:900; font-size:26px; margin:4px 0 0 0;}
.metric-sub{color:#475569; margin-top:6px; font-size:13px;}
</style>
""", unsafe_allow_html=True)


def card_open():
    st.markdown("<div class='bright-card'>", unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def bright_header(title: str, subtitle: str):
    st.markdown(f"<p class='title-grad'>{title}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtle'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# =========================
# Session state
# =========================
def init_state():
    if "menu" not in st.session_state:
        st.session_state.menu = "Home"
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "text_col" not in st.session_state:
        st.session_state.text_col = None
    if "prep_steps" not in st.session_state:
        st.session_state.prep_steps = {}
    if "final_df" not in st.session_state:
        st.session_state.final_df = None


init_state()


# =========================
# Utilities
# =========================
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


# =========================
# Scraping helper
# =========================
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


# =========================
# PREPROCESSING PIPELINE (aman)
# =========================
def case_folding(text: str) -> str:
    return to_text(text).lower()


def load_kamus_excel_safe(path: str) -> dict:
    try:
        df = pd.read_excel(path)
        # kolom yang umum
        if "non_standard" in df.columns and "standard_word" in df.columns:
            df = df[["non_standard", "standard_word"]].dropna()
            return dict(zip(df["non_standard"].astype(str), df["standard_word"].astype(str)))
        # fallback: kalau format beda, coba ambil 2 kolom pertama
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


# =========================
# LABELING (lexicon, aman + fallback)
# =========================
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


# =========================
# Sidebar navigation + Reset + Progress (FINAL)
# =========================
with st.sidebar:
    # ===== NAV =====
    st.markdown("### 🧭 Navigasi")

    def nav_button(label: str, menu_name: str, icon: str):
        is_active = (st.session_state.menu == menu_name)

        if is_active:
            st.markdown("<div class='nav-active'>", unsafe_allow_html=True)
        else:
            st.markdown("<div>", unsafe_allow_html=True)

        key_unique = f"navbtn_{menu_name}".replace(" ", "_")

        if st.button(f"{icon} {label}", use_container_width=True, key=key_unique):
            st.session_state.menu = menu_name
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
    nav_button("Home", "Home", "🏠")
    nav_button("Dataset", "Dataset", "📦")
    nav_button("Preprocessing", "Preprocessing", "🧼")
    nav_button("Klasifikasi SVM", "Klasifikasi SVM", "🧠")
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== PROGRESS (ringkas) =====
    st.markdown("---")
    step = 0
    if st.session_state.raw_df is not None:
        step = 1
    if st.session_state.final_df is not None:
        step = 2

    st.markdown("**📈 Progress**")
    st.progress(step / 2)
    st.caption(f"{step}/2 (Dataset → Preprocessing)")

    # ===== RESET (ringkas) =====
    if st.button("🔄 Reset", use_container_width=True, key="reset_sidebar_btn"):
        st.session_state.raw_df = None
        st.session_state.text_col = None
        st.session_state.prep_steps = {}
        st.session_state.final_df = None
        st.session_state.menu = "Home"
        st.rerun()


import html
import difflib

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
    """
    Highlight perbedaan kata:
    - kata yang dihapus: merah + strikethrough
    - kata yang ditambah: hijau
    Return HTML untuk before dan after.
    """
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
            # hanya ada di before
            b_out.extend([f"<span class='diff-del'>{t}</span>" for t in b_chunk])
        elif tag == "insert":
            # hanya ada di after
            a_out.extend([f"<span class='diff-add'>{t}</span>" for t in a_chunk])
        elif tag == "replace":
            b_out.extend([f"<span class='diff-del'>{t}</span>" for t in b_chunk])
            a_out.extend([f"<span class='diff-add'>{t}</span>" for t in a_chunk])

    return " ".join(b_out), " ".join(a_out)


def show_change_summary_and_examples(step_title: str, before_df: pd.DataFrame, after_df: pd.DataFrame, col: str = "content"):
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

    for idx, row in examples.iterrows():
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

# =========================
# MENU: HOME
# =========================
if st.session_state.menu == "Home":
    bright_header("💬 Sentimen Analyzer", "Ambil dataset → preprocessing bertahap → klasifikasi SVM (hasil langsung).")

    card_open()
    st.markdown(
        """
#### Apa yang bisa kamu lakukan?
- **Dataset**: scraping ulasan Google Play atau upload CSV/Excel.
- **Preprocessing**: tampil **bertahap** agar mudah dibandingkan.
- **Klasifikasi SVM**: lihat **confusion matrix, classification report, dan akurasi**.
        """.strip()
    )
    card_close()

    st.markdown("")
    if st.button("🚀 Mulai", use_container_width=True):
        st.session_state.menu = "Dataset"
        st.rerun()


# =========================
# MENU: DATASET
# =========================
elif st.session_state.menu == "Dataset":
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
        if st.button("➡️ Lanjut ke Preprocessing", use_container_width=True):
            st.session_state.menu = "Preprocessing"
            st.rerun()
        card_close()
    else:
        st.info("Silakan scraping atau upload dataset dulu.")






# =========================
# MENU: PREPROCESSING
# =========================


elif st.session_state.menu == "Preprocessing":
    bright_header("🧼 Preprocessing", "Klik tombol, lalu hasil tiap langkah akan muncul untuk dibandingkan.")

    if st.session_state.raw_df is None or not st.session_state.text_col:
        st.warning("Dataset/kolom teks belum siap. Kembali ke menu Dataset.")
        st.stop()

    ensure_nltk()

    # otomatis, tanpa input path
    ASSETS_DIR = "assets"
    KAMUS_PATH = os.path.join(ASSETS_DIR, "kamus.xlsx")
    LEX_POS_PATH = os.path.join(ASSETS_DIR, "positive.csv")
    LEX_NEG_PATH = os.path.join(ASSETS_DIR, "negative.csv")

    # status file (tanpa text_input)
    card_open()
    st.markdown("#### File pendukung (otomatis)")
    ok_kamus = os.path.exists(KAMUS_PATH)
    ok_pos = os.path.exists(LEX_POS_PATH)
    ok_neg = os.path.exists(LEX_NEG_PATH)
    st.write("📘 Kamus:", "✅ ditemukan" if ok_kamus else "⚠️ tidak ada (normalisasi dilewati)")
    st.write("🟢 Lexicon +:", "✅ ditemukan" if ok_pos else "⚠️ tidak ada (label fallback)")
    st.write("🔴 Lexicon -:", "✅ ditemukan" if ok_neg else "⚠️ tidak ada (label fallback)")
    card_close()

    drop_neutral = st.checkbox("Hapus data netral (score=0)", value=True)

    run_prep = st.button("⚙️ Jalankan Preprocessing", use_container_width=True)

    def show_compare(title, before_df, after_df, n=15):
        st.markdown("")
        card_open()
        st.markdown(f"### {title}")
    
        # ✅ Ringkasan + highlight contoh perubahan
        show_change_summary_and_examples(title, before_df, after_df, col="content")
    
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
        st.session_state.prep_steps["0) Data Awal"] = df0.copy()

        df1 = df0.copy()
        df1["content"] = df1["content"].apply(case_folding)
        st.session_state.prep_steps["1) Case Folding"] = df1.copy()

        kamus = load_kamus_excel_safe(KAMUS_PATH) if ok_kamus else {}
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
            # fallback aman
            df7["score"] = df7["tokens"].apply(lambda t: 1 if len(t) >= 5 else (-1 if 0 < len(t) < 3 else 0))
            df7["Sentimen"] = df7["score"].apply(lambda s: "positif" if s > 0 else ("negatif" if s < 0 else "netral"))
        
        # ✅ SIMPAN distribusi sebelum filter netral
        dist_sebelum = df7["Sentimen"].value_counts()
        st.session_state.label_dist_before = dist_sebelum.rename_axis("Label").reset_index(name="Jumlah")
        st.session_state.label_counts_before = {
            "positif": int(dist_sebelum.get("positif", 0)),
            "negatif": int(dist_sebelum.get("negatif", 0)),
            "netral": int(dist_sebelum.get("netral", 0)),
        }
        
        # ✅ BARU FILTER NETRAL (JIKA DIPILIH)
        if drop_neutral:
            df7 = df7[df7["Sentimen"] != "netral"].reset_index(drop=True)
        
        st.session_state.prep_steps["7) Pelabelan"] = df7.copy()
        st.session_state.final_df = df7.copy()

        st.success("Preprocessing selesai! Scroll untuk melihat perbandingan.")

    # tampilkan hasil bertahap (kalau sudah ada)
    steps = st.session_state.prep_steps
    if steps:
        keys = list(steps.keys())
        for i in range(1, len(keys)):
            before = steps[keys[i - 1]]
            after = steps[keys[i]]
    
            if keys[i].startswith("6) Tokenizing"):
                st.markdown("")
                card_open()
                st.markdown("### 6) Tokenizing")
    
                st.markdown("**Contoh hasil token (teks → tokens):**")
                st.dataframe(after[["content", "tokens"]].head(12), use_container_width=True)
    
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Sebelum (teks)**")
                    st.dataframe(before[["content"]].head(15), use_container_width=True)
                with c2:
                    st.markdown("**Sesudah (tokens)**")
                    st.dataframe(after[["content", "tokens"]].head(15), use_container_width=True)
                card_close()
    
            elif keys[i].startswith("7) Pelabelan"):
                st.markdown("")
                card_open()
                st.markdown("### 7) Pelabelan Sentimen")
            
                # ✅ INFO SEBELUM FILTER NETRAL (tampilan seperti screenshot)
                counts = st.session_state.get("label_counts_before", None)
                dist_before_df = st.session_state.get("label_dist_before", None)
            
                if counts is not None:
                    st.markdown("#### 📌 Info sebelum filter netral")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Positif", counts.get("positif", 0))
                    c2.metric("Negatif", counts.get("negatif", 0))
                    c3.metric("Netral (sebelum filter)", counts.get("netral", 0))
            
                    if dist_before_df is not None:
                        st.dataframe(dist_before_df, use_container_width=True)
                else:
                    st.info("Info sebelum filter belum tersedia. Jalankan preprocessing dulu.")
            
                st.markdown("---")
            
                # ✅ Distribusi label SETELAH filter (yang sekarang kamu tampilkan)
                st.markdown("**Distribusi label (setelah filter):**")
                dist_after = after["Sentimen"].value_counts().rename_axis("Label").reset_index(name="Jumlah")
                st.dataframe(dist_after, use_container_width=True)
            
                st.markdown("---")
                st.markdown("**Contoh hasil pelabelan:**")
                st.dataframe(after[["content", "tokens", "score", "Sentimen"]].head(25), use_container_width=True)
                card_close()

    
            else:
                show_compare(keys[i], before, after)
    
        st.markdown("")
        if st.button("➡️ Lanjut ke Klasifikasi SVM", use_container_width=True):
            st.session_state.menu = "Klasifikasi SVM"
            st.rerun()
    else:
        st.info("Klik tombol 'Jalankan Preprocessing' untuk memulai.")



# =========================
# MENU: KLASIFIKASI SVM (versi awam-friendly)
# =========================
elif st.session_state.menu == "Klasifikasi SVM":
    bright_header("🧠 Klasifikasi SVM", "Hasil dibuat lebih mudah dipahami untuk orang awam.")

    if st.session_state.final_df is None:
        st.warning("Data belum siap. Jalankan preprocessing dulu.")
        st.stop()

    df = st.session_state.final_df.copy()
    if "content" not in df.columns or "Sentimen" not in df.columns:
        st.error("Kolom wajib tidak ada: butuh 'content' dan 'Sentimen'")
        st.stop()

    # Pastikan hanya 2 kelas
    df = df[df["Sentimen"].isin(["positif", "negatif"])].copy()
    df["content"] = df["content"].astype(str).fillna("")

    card_open()
    st.markdown("#### Jalankan SVM (TF-IDF, split 80/20)")
    run_svm = st.button("🚀 Mulai Klasifikasi SVM", use_container_width=True)
    card_close()

    if not run_svm:
        st.info("Klik tombol di atas untuk melihat hasil model.")
        st.stop()

    # ======================
    # TRAIN + PREDICT
    # ======================
    X = df["content"]
    y = df["Sentimen"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LinearSVC()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)

    labels = ["negatif", "positif"]  # urutan tetap agar CM konsisten
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # ======================
    # NARASI AWAM
    # ======================
    total = cm.sum()
    benar = int(cm[0, 0] + cm[1, 1])
    salah = int(cm[0, 1] + cm[1, 0])

    salah_neg_jadi_pos = int(cm[0, 1])
    salah_pos_jadi_neg = int(cm[1, 0])

    # kalimat kesimpulan
    kesimpulan = f"Model benar menebak **{benar} dari {total}** ulasan (≈ **{acc*100:.1f}%**)."
    if salah_neg_jadi_pos > salah_pos_jadi_neg:
        kesimpulan2 = f"Kesalahan yang paling sering: **ulasan negatif dikira positif** ({salah_neg_jadi_pos} kasus)."
    elif salah_pos_jadi_neg > salah_neg_jadi_pos:
        kesimpulan2 = f"Kesalahan yang paling sering: **ulasan positif dikira negatif** ({salah_pos_jadi_neg} kasus)."
    else:
        kesimpulan2 = f"Kesalahan negatif→positif dan positif→negatif jumlahnya mirip."

    # ======================
    # TAMPILKAN RINGKASAN
    # ======================
    st.markdown("")
    card_open()
    st.markdown("### ✅ Ringkasan Hasil")
    st.markdown(kesimpulan)
    st.markdown(kesimpulan2)
    st.markdown(
        "- **Benar** artinya prediksi sama dengan label asli.\n"
        "- **Salah** artinya prediksi berbeda dari label asli."
    )
    card_close()

    # ======================
    # CONFUSION MATRIX versi awam: 4 kartu
    # ======================
    st.markdown("")
    card_open()
    st.markdown("### 🔎 Confusion Matrix (versi mudah)")

    a, b, c, d = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    # a: negatif->negatif, b: negatif->positif, c: positif->negatif, d: positif->positif

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

    st.markdown("---")
    st.caption("Catatan: 4 kartu di atas adalah bentuk yang sama dengan confusion matrix, hanya dibuat lebih mudah dibaca.")
    card_close()

    # ======================
    # PENJELASAN METRIK (awam-friendly)
    # ======================
    st.markdown("")
    card_open()
    st.markdown("### 📘 Apa arti Precision / Recall / F1?")

    with st.expander("Klik untuk lihat penjelasan sederhana"):
        st.markdown(
            """
- **Precision (ketepatan)**: Kalau model bilang “positif”, seberapa sering itu benar?
- **Recall (kelengkapan)**: Dari semua yang benar-benar “positif”, seberapa banyak yang berhasil ditangkap model?
- **F1-Score**: Nilai ringkasan yang menyeimbangkan precision dan recall.
- **Support**: Jumlah data (ulasan) pada kelas tersebut.
            """.strip()
        )

    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).transpose()
    st.dataframe(rep_df, use_container_width=True)
    card_close()

    # ======================
    # CONTOH SALAH PREDIKSI (paling penting untuk awam)
    # ======================
    st.markdown("")
    card_open()
    st.markdown("### ❗ Contoh Ulasan yang Salah Prediksi")

    # Buat dataframe evaluasi
    eval_df = pd.DataFrame({
        "Ulasan": X_test.values,
        "Label Asli": y_test.values,
        "Prediksi Model": y_pred
    })

    wrong_df = eval_df[eval_df["Label Asli"] != eval_df["Prediksi Model"]].copy()
    if wrong_df.empty:
        st.success("Tidak ada salah prediksi pada data uji (jarang terjadi, tapi bisa).")
    else:
        st.caption("Berikut beberapa contoh yang membuat model keliru (ini membantu orang awam memahami batasan model).")
        st.dataframe(wrong_df.head(15), use_container_width=True)
    card_close()

    # ======================
    # CONTOH PALING “YAKIN” (pakai decision_function)
    # ======================
    st.markdown("")
    card_open()
    st.markdown("### ⭐ Contoh Prediksi Paling Yakin (agar terasa nyata)")

    try:
        scores = model.decision_function(X_test_vec)
        # Binary: score > 0 biasanya ke kelas "positif" tergantung urutan training
        # Kita tetap pakai skor absolut untuk confidence
        eval_df["Skor Keyakinan"] = np.abs(scores)

        top_conf = eval_df.sort_values("Skor Keyakinan", ascending=False).head(10)
        st.caption("Semakin besar skor keyakinan, semakin yakin model dengan prediksinya.")
        st.dataframe(top_conf, use_container_width=True)
    except Exception:
        st.info("Model tidak menyediakan skor keyakinan untuk ditampilkan pada konfigurasi ini.")

    card_close()

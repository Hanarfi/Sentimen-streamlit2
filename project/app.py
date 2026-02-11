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
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}

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

True)



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
# Sidebar navigation
# =========================
with st.sidebar:
    def go(menu_name: str):
        st.session_state.menu = menu_name
        st.rerun()
    
    def nav_button(label: str, menu_name: str, icon: str, badge: str = ""):
        is_active = (st.session_state.menu == menu_name)
    
        # wrapper untuk menandai active state via CSS class
        if is_active:
            st.markdown("<div class='nav-active'>", unsafe_allow_html=True)
        else:
            st.markdown("<div>", unsafe_allow_html=True)
    
        # label dengan badge (opsional)
        nice_label = f"{icon} {label}"
        if badge:
            nice_label = f"{nice_label} <span class='nav-badge'>{badge}</span>"
    
        if st.button(nice_label, use_container_width=True, key=f"nav_{menu_name}"):
            go(menu_name)
    
        st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("### 🧭 Navigasi")
    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
    
    nav_button("Home", "Home", "🏠")
    nav_button("Dataset", "Dataset", "📦", badge="1")
    nav_button("Preprocessing", "Preprocessing", "🧼", badge="2")
    nav_button("Klasifikasi SVM", "Klasifikasi SVM", "🧠", badge="3")
    
    st.markdown("</div>", unsafe_allow_html=True)

    
    
    st.markdown("### 🧭 Navigasi")
    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
      
    nav_button("Home", "Home", "🏠")
    nav_button("Dataset", "Dataset", "📦", badge="1")
    nav_button("Preprocessing", "Preprocessing", "🧼", badge="2")
    nav_button("Klasifikasi SVM", "Klasifikasi SVM", "🧠", badge="3")
      
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔄 Reset")

if st.button("🧹 Reset & Kembali ke Home", use_container_width=True):
    st.session_state.raw_df = None
    st.session_state.text_col = None
    st.session_state.prep_steps = {}
    st.session_state.final_df = None
    st.session_state.menu = "Home"
    st.rerun()

st.markdown("---")
st.markdown("### 📈 Progress")

step = 0
if st.session_state.raw_df is not None:
    step = 1
if st.session_state.final_df is not None:
    step = 2
st.progress(step / 2)
st.caption(f"Langkah selesai: {step}/2 (Dataset → Preprocessing)")



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
        df2["content"] = df2["content"].apply(lambda x: normalisasi_kamus(x, kamus))
        st.session_state.prep_steps["2) Normalisasi"] = df2.copy()

        df3 = df2.copy()
        df3["content"] = df3["content"].apply(data_cleansing)
        st.session_state.prep_steps["3) Data Cleansing"] = df3.copy()

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
                st.dataframe(after[["content", "tokens", "score", "Sentimen"]].head(30), use_container_width=True)
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
# MENU: KLASIFIKASI SVM
# =========================
elif st.session_state.menu == "Klasifikasi SVM":
    bright_header("🧠 Klasifikasi SVM", "Klik tombol untuk melihat confusion matrix, report, dan akurasi.")

    if st.session_state.final_df is None:
        st.warning("Data belum siap. Jalankan preprocessing dulu.")
        st.stop()

    df = st.session_state.final_df.copy()
    if "content" not in df.columns or "Sentimen" not in df.columns:
        st.error("Kolom wajib tidak ada: butuh 'content' dan 'Sentimen'")
        st.stop()

    card_open()
    st.markdown("#### Jalankan SVM (TF-IDF, split 80/20)")
    run_svm = st.button("🚀 Mulai Klasifikasi SVM", use_container_width=True)
    card_close()

    if run_svm:
        X = df["content"].astype(str).fillna("")
        y = df["Sentimen"].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if len(y.unique()) > 1 else None
        )

        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        model = LinearSVC()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        st.markdown("")
        card_open()
        st.markdown(f"### ✅ Akurasi: `{acc:.4f}`")
        card_close()

        st.markdown("")
        card_open()
        st.markdown("### Confusion Matrix")
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.yticks(range(len(labels)), labels)

        thresh = cm.max() / 2 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
        plt.tight_layout()
        st.pyplot(fig)
        card_close()

        st.markdown("")
        card_open()
        st.markdown("### Classification Report")
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)
        card_close()

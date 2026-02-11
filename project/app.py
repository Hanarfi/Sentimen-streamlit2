import os
import re
import csv
import numpy as np
import pandas as pd
import streamlit as st

# ===== Scraping =====
from google_play_scraper import reviews, Sort

# ===== NLP =====
import emoji
import nltk
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ===== ML =====
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import matplotlib.pyplot as plt


# =========================
# Streamlit Config + Light UI
# =========================
st.set_page_config(page_title="Sentimen Analyzer", page_icon="💬", layout="wide")

BRIGHT_CSS = """
<style>
.stApp {background: linear-gradient(180deg,#F7FBFF 0%,#FFFFFF 60%,#F7FFFB 100%);}
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.bright-card{
  background:white; border-radius:18px; padding:18px;
  border:1px solid #E9F2FF; box-shadow:0 8px 22px rgba(25,118,210,0.08);
}
.title-grad{
  font-size:34px; font-weight:800;
  background:linear-gradient(90deg,#1565C0,#00BFA5);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin:0; line-height:1.1;
}
.subtle{color:#5F6B7A; font-size:14px;}
.divider{height:1px; background:#ECF3FF; margin:12px 0 18px 0;}
</style>
"""
st.markdown(BRIGHT_CSS, unsafe_allow_html=True)


def card_open():
    st.markdown("<div class='bright-card'>", unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def bright_header(title: str, subtitle: str):
    st.markdown(f"<p class='title-grad'>{title}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtle'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# =========================
# Session State
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
# NLTK resources safe
# =========================
def ensure_nltk():
    try:
        _ = stopwords.words("indonesian")
    except LookupError:
        nltk.download("stopwords")


# =========================
# Scraping (simple)
# =========================
def scrape_google_play(app_id: str, jumlah: int, bahasa="id", negara="id"):
    all_rows = []
    count = 0
    next_token = None

    while count < jumlah:
        batch, next_token = reviews(
            app_id,
            lang=bahasa,
            country=negara,
            sort=Sort.NEWEST,
            count=min(200, jumlah - count),
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
        count = len(all_rows)
        if next_token is None:
            break

    return pd.DataFrame(all_rows)


# =========================
# PREPROCESSING SAFE FUNCTIONS
# (robust & minim error)
# =========================
@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


def to_text(x) -> str:
    # aman untuk NaN / None
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def case_folding(text: str) -> str:
    return to_text(text).lower()


def load_kamus_excel_safe(path: str) -> dict:
    """
    Aman: kalau file tidak ada / format tidak cocok, return dict kosong.
    """
    try:
        df = pd.read_excel(path)
        # coba cari kolom yang sesuai
        possible_nonstd = ["non_standard", "nonstandard", "slang", "kata_tidak_baku"]
        possible_std = ["standard_word", "standard", "kata_baku"]

        col_nonstd = None
        col_std = None
        for c in possible_nonstd:
            if c in df.columns:
                col_nonstd = c
                break
        for c in possible_std:
            if c in df.columns:
                col_std = c
                break

        # fallback: kalau di file kamu memang "non_standard" dan "standard_word"
        if col_nonstd is None and "non_standard" in df.columns:
            col_nonstd = "non_standard"
        if col_std is None and "standard_word" in df.columns:
            col_std = "standard_word"

        if col_nonstd is None or col_std is None:
            return {}

        df = df[[col_nonstd, col_std]].dropna()
        return dict(zip(df[col_nonstd].astype(str), df[col_std].astype(str)))
    except Exception:
        return {}


def normalisasi_kamus(text: str, kamus: dict) -> str:
    """
    Aman: kalau kamus kosong, tidak mengubah teks.
    """
    text = to_text(text)
    if not kamus:
        return text
    words = text.split()
    out = [kamus.get(w, w) for w in words]
    return " ".join(out)


def data_cleansing(text: str) -> str:
    """
    Versi aman & sederhana:
    - buang URL, mention, hashtag, angka
    - buang emoji
    - sisakan a-z dan spasi
    - rapikan spasi
    """
    text = to_text(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_id(text: str) -> str:
    text = to_text(text)
    sw = set(stopwords.words("indonesian"))
    tokens = text.split()
    tokens = [t for t in tokens if t not in sw]
    return " ".join(tokens)


def stemming_sastrawi(text: str) -> str:
    text = to_text(text)
    stemmer = get_stemmer()
    return stemmer.stem(text)


def tokenizing_simple(text: str):
    """
    Tidak pakai nltk.word_tokenize agar minim error dependency.
    Tokenisasi cukup split spasi (untuk app awam ini sudah oke).
    """
    text = to_text(text).strip()
    if not text:
        return []
    return text.split()


# =========================
# LABELING SAFE (Lexicon-based)
# =========================
def load_lexicon_safe(path: str) -> dict:
    """
    Format yang didukung:
    - CSV: kata, skor
    Aman jika ada header / baris kotor.
    """
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
                    # skip jika skor bukan angka
                    pass
    except Exception:
        pass
    return lex


def label_by_lexicon(tokens, lex_pos: dict, lex_neg: dict):
    """
    Skor = jumlah bobot kata positif - negatif.
    Label:
    - score > 0 => positif
    - score < 0 => negatif
    - else => netral
    Aman untuk token kosong.
    """
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
# Sidebar Menu
# =========================
with st.sidebar:
    st.markdown("### 🧭 Navigasi")
    menu = st.radio(
        "Pilih menu",
        ["Home", "Dataset", "Preprocessing", "Klasifikasi SVM"],
        index=["Home", "Dataset", "Preprocessing", "Klasifikasi SVM"].index(st.session_state.menu),
    )
    st.session_state.menu = menu

    st.markdown("---")
    st.markdown("### ✅ Status")
    if st.session_state.raw_df is None:
        st.warning("Dataset belum ada")
    else:
        st.success(f"Dataset: {len(st.session_state.raw_df)} baris")

    if st.session_state.text_col:
        st.info(f"Kolom teks: {st.session_state.text_col}")
    else:
        st.info("Kolom teks: belum dipilih")

    if st.session_state.final_df is not None:
        st.success("Data siap untuk SVM")


# =========================
# HOME
# =========================
if st.session_state.menu == "Home":
    bright_header(
        "💬 Sentimen Analyzer",
        "Untuk orang awam: ambil dataset → preprocessing bertahap → klasifikasi SVM.",
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        card_open()
        st.markdown("#### Tentang aplikasi ini")
        st.markdown(
            """
Aplikasi ini membantu kamu melakukan:
- **Scraping ulasan Google Play** atau **upload dataset**,
- Memilih kolom teks,
- **Preprocessing bertahap** (kamu bisa bandingkan hasil tiap tahap),
- **Pelabelan sentimen** berbasis lexicon,
- **Klasifikasi SVM** (TF-IDF, split 80/20) dan lihat hasilnya.
            """.strip()
        )
        card_close()

        if st.button("🚀 Mulai", use_container_width=True):
            st.session_state.menu = "Dataset"
            st.rerun()

    with col2:
        card_open()
        st.markdown("#### Catatan")
        st.markdown(
            """
- Untuk preprocessing & pelabelan, file lexicon/kamus **opsional**.
- Jika file tidak ada, preprocessing tetap jalan (normalisasi dilewati, pelabelan pakai aturan sederhana).
            """.strip()
        )
        card_close()


# =========================
# DATASET
# =========================
elif st.session_state.menu == "Dataset":
    bright_header(
        "📦 Dataset",
        "Pilih sumber dataset: scraping Google Play atau upload CSV/Excel. Lalu pilih kolom teks.",
    )

    tab1, tab2 = st.tabs(["🕷️ Scraping Google Play", "📤 Upload Dataset"])

    with tab1:
        card_open()
        st.markdown("#### Scraping super sederhana")
        st.caption("Masukkan App ID → jumlah ulasan → klik tombol.")
        app_id = st.text_input("App ID Google Play", placeholder="contoh: com.whatsapp")
        jumlah = st.number_input("Jumlah ulasan", min_value=50, max_value=5000, value=200, step=50)
        bahasa = st.selectbox("Bahasa", ["id", "en"], index=0)
        negara = st.selectbox("Negara", ["id", "us", "sg", "my"], index=0)

        cA, cB = st.columns(2)
        with cA:
            do_scrape = st.button("🧲 Mulai Scraping", use_container_width=True)
        with cB:
            if st.button("🧹 Reset Dataset", use_container_width=True):
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
        st.markdown("#### Upload CSV / Excel")
        file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
        if file is not None:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                st.session_state.raw_df = df
                # auto-guess kolom teks
                cols = list(df.columns)
                guess = "content" if "content" in cols else cols[0]
                st.session_state.text_col = guess

                st.session_state.prep_steps = {}
                st.session_state.final_df = None
                st.success(f"Dataset berhasil di-load: {len(df)} baris, {df.shape[1]} kolom.")
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
        card_close()

    if st.session_state.raw_df is not None:
        st.markdown("")
        card_open()
        st.markdown("#### Preview Dataset")
        st.dataframe(st.session_state.raw_df.head(50), use_container_width=True)
        card_close()

        st.markdown("")
        card_open()
        st.markdown("#### Pilih kolom teks")
        cols = list(st.session_state.raw_df.columns)
        idx = cols.index(st.session_state.text_col) if st.session_state.text_col in cols else 0
        st.session_state.text_col = st.selectbox("Kolom yang dianalisis", cols, index=idx)

        if st.button("➡️ Lanjut ke Preprocessing", use_container_width=True):
            st.session_state.menu = "Preprocessing"
            st.rerun()
        card_close()
    else:
        st.info("Silakan scraping atau upload dataset dulu.")


# =========================
# PREPROCESSING
# =========================
elif st.session_state.menu == "Preprocessing":
    bright_header(
        "🧼 Preprocessing",
        "Klik tombol, lalu hasil tiap langkah akan muncul agar mudah dibandingkan.",
    )

    if st.session_state.raw_df is None or not st.session_state.text_col:
        st.warning("Dataset/kolom teks belum siap. Kembali ke menu Dataset.")
        st.stop()

    ensure_nltk()

    ASSETS_DIR = "assets"
    kamus_path = os.path.join(ASSETS_DIR, "kamus.xlsx")  # boleh kamu rename bebas
    lex_pos_path = os.path.join(ASSETS_DIR, "positive.csv")
    lex_neg_path = os.path.join(ASSETS_DIR, "negative.csv")

    colL, colR = st.columns([1.1, 1])
    with colL:
        card_open()
        st.markdown("#### Pengaturan file (opsional)")
        st.caption("Kalau file tidak ada, aplikasi tetap jalan (normalisasi dilewati, lexicon fallback).")
        kamus_path = st.text_input("Path Kamus Excel (opsional)", value=kamus_path)
        lex_pos_path = st.text_input("Path Lexicon Positif (opsional)", value=lex_pos_path)
        lex_neg_path = st.text_input("Path Lexicon Negatif (opsional)", value=lex_neg_path)
        card_close()

    with colR:
        card_open()
        drop_neutral = st.checkbox("Hapus data netral (score = 0)", value=True)
        st.caption("Jika dicentang, hanya positif & negatif yang lanjut ke SVM.")
        card_close()

    st.markdown("")
    run_prep = st.button("⚙️ Jalankan Preprocessing", use_container_width=True)

    def show_compare(step_name, before_df, after_df, preview=15):
        st.markdown("")
        card_open()
        st.markdown(f"### {step_name}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sebelum**")
            st.dataframe(before_df[["content"]].head(preview), use_container_width=True)
        with c2:
            st.markdown("**Sesudah**")
            st.dataframe(after_df[["content"]].head(preview), use_container_width=True)
        card_close()

    if run_prep:
        base = st.session_state.raw_df.copy()
        text_col = st.session_state.text_col

        df0 = base.copy()
        df0["content"] = df0[text_col].apply(to_text)

        st.session_state.prep_steps = {}
        st.session_state.final_df = None

        st.session_state.prep_steps["0) Data Awal"] = df0.copy()

        # 1) Case folding
        df1 = df0.copy()
        df1["content"] = df1["content"].apply(case_folding)
        st.session_state.prep_steps["1) Case Folding"] = df1.copy()

        # 2) Normalisasi (kamus) - aman jika file tidak ada
        kamus = load_kamus_excel_safe(kamus_path) if os.path.exists(kamus_path) else {}
        df2 = df1.copy()
        df2["content"] = df2["content"].apply(lambda x: normalisasi_kamus(x, kamus))
        st.session_state.prep_steps["2) Normalisasi (Kamus)"] = df2.copy()

        # 3) Cleansing
        df3 = df2.copy()
        df3["content"] = df3["content"].apply(data_cleansing)
        st.session_state.prep_steps["3) Data Cleansing"] = df3.copy()

        # 4) Stopword removal
        df4 = df3.copy()
        df4["content"] = df4["content"].apply(remove_stopwords_id)
        st.session_state.prep_steps["4) Stopword Removal"] = df4.copy()

        # 5) Stemming
        df5 = df4.copy()
        df5["content"] = df5["content"].apply(stemming_sastrawi)
        st.session_state.prep_steps["5) Stemming"] = df5.copy()

        # 6) Tokenizing
        df6 = df5.copy()
        df6["tokens"] = df6["content"].apply(tokenizing_simple)
        st.session_state.prep_steps["6) Tokenizing"] = df6.copy()

        # 7) Labeling (lexicon) - aman jika file tidak ada
        lex_pos = load_lexicon_safe(lex_pos_path) if os.path.exists(lex_pos_path) else {}
        lex_neg = load_lexicon_safe(lex_neg_path) if os.path.exists(lex_neg_path) else {}

        df7 = df6.copy()
        if lex_pos or lex_neg:
            # lexicon-based
            res = df7["tokens"].apply(lambda toks: label_by_lexicon(toks, lex_pos, lex_neg))
            df7["score"] = res.apply(lambda x: x[0])
            df7["Sentimen"] = res.apply(lambda x: x[1])
        else:
            # fallback sederhana agar tidak error: panjang token -> label
            df7["score"] = df7["tokens"].apply(lambda t: 1 if len(t) >= 5 else (-1 if 0 < len(t) < 3 else 0))
            df7["Sentimen"] = df7["score"].apply(lambda s: "positif" if s > 0 else ("negatif" if s < 0 else "netral"))

        if drop_neutral:
            df7 = df7[df7["Sentimen"] != "netral"].reset_index(drop=True)

        st.session_state.prep_steps["7) Pelabelan"] = df7.copy()
        st.session_state.final_df = df7.copy()

        st.success("Preprocessing selesai! Scroll ke bawah untuk membandingkan tiap tahap.")

    # tampilkan hasil bertahap
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
# KLASIFIKASI SVM
# =========================
elif st.session_state.menu == "Klasifikasi SVM":
    bright_header(
        "🧠 Klasifikasi SVM",
        "Klik tombol untuk melihat confusion matrix, classification report, dan akurasi.",
    )

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

        # Split 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if len(y.unique()) > 1 else None
        )

        # TF-IDF (tidak ditampilkan)
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        # SVM
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

        # Confusion matrix
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

        # Classification report
        st.markdown("")
        card_open()
        st.markdown("### Classification Report")
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).transpose(), use_container_width=True)
        card_close()

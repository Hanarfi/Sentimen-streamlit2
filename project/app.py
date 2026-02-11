import os
import re
import ast
import csv
import numpy as np
import pandas as pd
import streamlit as st

# ===== Scraping Google Play =====
from google_play_scraper import reviews, Sort

# ===== NLP =====
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ===== ML =====
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import matplotlib.pyplot as plt


# =========================
# Page Config + Bright UI
# =========================
st.set_page_config(
    page_title="Sentimen Analyzer (Streamlit)",
    page_icon="💬",
    layout="wide",
)

BRIGHT_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #F8FBFF 0%, #FFFFFF 60%, #F7FFFB 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .bright-card {
        background: white;
        border-radius: 18px;
        padding: 18px 18px;
        border: 1px solid #E9F2FF;
        box-shadow: 0 8px 22px rgba(25, 118, 210, 0.08);
    }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #E8F3FF;
        color: #0D47A1;
        font-weight: 600;
        font-size: 12px;
        margin-right: 6px;
    }
    .title-grad {
        font-size: 34px;
        font-weight: 800;
        background: linear-gradient(90deg, #1565C0, #00BFA5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }
    .subtle {
        color: #5F6B7A;
        font-size: 14px;
    }
    .btn-hint {
        color: #5F6B7A;
        font-size: 13px;
        margin-top: -8px;
    }
    .divider {
        height: 1px;
        background: #ECF3FF;
        margin: 12px 0 18px 0;
    }
</style>
"""
st.markdown(BRIGHT_CSS, unsafe_allow_html=True)


# =========================
# Helpers: session init
# =========================
def init_state():
    if "menu" not in st.session_state:
        st.session_state.menu = "Home"

    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None  # dataset awal (setelah scraping/upload)
    if "text_col" not in st.session_state:
        st.session_state.text_col = None  # kolom yang dipilih untuk preprocessing

    if "prep_steps" not in st.session_state:
        # dict: step_name -> dataframe hasil step
        st.session_state.prep_steps = {}

    if "final_df" not in st.session_state:
        st.session_state.final_df = None  # dataset final setelah pelabelan (siap SVM)


init_state()


# =========================
# NLTK ensure resources
# =========================
def ensure_nltk():
    try:
        _ = stopwords.words("indonesian")
    except LookupError:
        nltk.download("stopwords")
    try:
        _ = word_tokenize("tes")
    except LookupError:
        nltk.download("punkt")


# =========================
# Preprocessing functions (acuannya dari kamu)
# =========================
def case_folding(text: str) -> str:
    return str(text).lower()


def load_kamus_excel(kamus_path: str) -> dict:
    baca_kamus = pd.read_excel(kamus_path)
    kamus_dict = dict(zip(baca_kamus["non_standard"], baca_kamus["standard_word"]))
    return kamus_dict


def normalisasi_dengan_kamus(text: str, kamus_dict: dict) -> str:
    words = str(text).split()
    normalized_words = []
    for word in words:
        if word in kamus_dict:
            normalized_words.append(kamus_dict[word])
        else:
            normalized_words.append(word)
    return " ".join(normalized_words)


def data_cleansing(text: str) -> str:
    text = str(text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"#[A-Za-z0-9]+", "", text)
    text = re.sub(r"RT[\s]", "", text)
    text = re.sub(r"RT[?|$|.|@!&:_=)(><,]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]", "", text)
    text = text.replace("\n", " ").strip(" ")
    text = re.sub("s://t.co/", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.replace('"', "")
    # hanya huruf a-z dan spasi (sesuai kode kamu)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)  # reduksi huruf berulang
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_id(text: str) -> str:
    stop_words = set(stopwords.words("indonesian"))
    tokens = str(text).split()
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)


@st.cache_resource
def get_sastrawi_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


def stem_text(text: str) -> str:
    stemmer = get_sastrawi_stemmer()
    return stemmer.stem(str(text))


def load_lexicon(path: str) -> dict:
    lex = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if len(row) >= 2:
                try:
                    lex[row[0].strip()] = int(row[1])
                except:
                    pass
    return lex


def filter_tokens_by_lexicon(tokens, lex_pos: dict, lex_neg: dict):
    if not isinstance(tokens, list):
        return []
    return [w for w in tokens if (w in lex_pos) or (w in lex_neg)]


def sentiment_analysis_lexicon_indonesia(tokens, lex_pos: dict, lex_neg: dict):
    score = 0
    for w in tokens:
        if w in lex_pos:
            score += lex_pos[w]
        if w in lex_neg:
            score += lex_neg[w]

    if score > 0:
        sent = "positif"
    elif score < 0:
        sent = "negatif"
    else:
        sent = "netral"
    return score, sent


# =========================
# Scraping helper
# =========================
def scrape_google_play(app_id: str, jumlah: int, bahasa="id", negara="id"):
    """
    Scraping ulasan Google Play super sederhana untuk user awam:
    - input app_id
    - input jumlah ulasan
    """
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
# UI pieces
# =========================
def bright_header(title: str, subtitle: str):
    st.markdown(f"<p class='title-grad'>{title}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtle'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


def card_open():
    st.markdown("<div class='bright-card'>", unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


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
    st.markdown("### ✅ Status Data")
    if st.session_state.raw_df is None:
        st.warning("Dataset belum ada.")
    else:
        st.success(f"Dataset siap: {len(st.session_state.raw_df)} baris")

    if st.session_state.text_col:
        st.info(f"Kolom teks: **{st.session_state.text_col}**")
    else:
        st.info("Kolom teks: belum dipilih")

    if st.session_state.final_df is not None:
        st.success("Preprocessing + pelabelan: siap untuk SVM")


# =========================
# MENU: HOME
# =========================
if st.session_state.menu == "Home":
    bright_header(
        "💬 Sentimen Analyzer",
        "Aplikasi sederhana untuk scraping / input dataset, preprocessing bertahap, dan klasifikasi SVM.",
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        card_open()
        st.markdown("#### Apa yang bisa dilakukan aplikasi ini?")
        st.markdown(
            """
- **Ambil data ulasan** dari Google Play (scraping sederhana), atau **upload dataset sendiri**.
- Pilih **kolom teks** yang ingin dianalisis.
- Jalankan **preprocessing bertahap** (dibandingkan satu per satu).
- Buat **label sentimen** menggunakan **lexicon**.
- Jalankan **SVM + TF-IDF** dan langsung lihat hasil: **confusion matrix, classification report, akurasi**.
            """.strip()
        )
        card_close()

        st.markdown("")
        if st.button("🚀 Mulai", use_container_width=True):
            st.session_state.menu = "Dataset"
            st.rerun()

    with col2:
        card_open()
        st.markdown("#### Tips untuk pengguna awam")
        st.markdown(
            """
- Mulai dari menu **Dataset** → pilih **Scraping** atau **Upload file CSV/Excel**.
- Pastikan kolom teks berisi kalimat ulasan (contoh: `content`).
- Di menu **Preprocessing**, kamu akan melihat perubahan teks **tahap demi tahap**.
- Di menu **Klasifikasi SVM**, kamu tinggal klik tombol untuk melihat hasil.
            """.strip()
        )
        st.markdown("<span class='pill'>UI cerah</span><span class='pill'>Langkah jelas</span><span class='pill'>Fokus hasil</span>", unsafe_allow_html=True)
        card_close()


# =========================
# MENU: DATASET
# =========================
elif st.session_state.menu == "Dataset":
    bright_header(
        "📦 Dataset",
        "Pilih sumber data: scraping Google Play atau upload dataset. Setelah itu pilih kolom teks untuk preprocessing.",
    )

    tab1, tab2 = st.tabs(["🕷️ Scraping Google Play", "📤 Upload Dataset"])

    with tab1:
        card_open()
        st.markdown("#### Scraping yang sederhana (untuk orang awam)")
        st.markdown(
            """
1) Masukkan **App ID** Google Play (contoh: `com.bca` atau `com.whatsapp`)  
2) Masukkan **jumlah ulasan** yang ingin diambil  
3) Klik **Mulai Scraping**  
            """.strip()
        )

        app_id = st.text_input("App ID Google Play", value="", placeholder="contoh: com.whatsapp")
        jumlah = st.number_input("Jumlah ulasan", min_value=50, max_value=5000, value=200, step=50)
        bahasa = st.selectbox("Bahasa", ["id", "en"], index=0)
        negara = st.selectbox("Negara", ["id", "us", "sg", "my"], index=0)

        colA, colB = st.columns([1, 1])
        with colA:
            do_scrape = st.button("🧲 Mulai Scraping", use_container_width=True)
        with colB:
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
                with st.spinner("Mengambil ulasan dari Google Play..."):
                    df = scrape_google_play(app_id.strip(), int(jumlah), bahasa=bahasa, negara=negara)
                if df.empty:
                    st.warning("Tidak ada data yang berhasil diambil. Coba app_id lain atau jumlah lebih kecil.")
                else:
                    st.session_state.raw_df = df
                    st.success(f"Berhasil mengambil {len(df)} ulasan.")
        card_close()

        if st.session_state.raw_df is not None:
            st.markdown("")
            card_open()
            st.markdown("#### Preview Dataset")
            st.dataframe(st.session_state.raw_df.head(50), use_container_width=True)
            card_close()

    with tab2:
        card_open()
        st.markdown("#### Upload dataset (CSV / Excel)")
        file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
        if file is not None:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                st.session_state.raw_df = df
                st.session_state.text_col = None
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

    # pilih kolom teks untuk preprocessing
    st.markdown("")
    if st.session_state.raw_df is not None:
        card_open()
        st.markdown("#### Pilih kolom teks untuk preprocessing")
        cols = list(st.session_state.raw_df.columns)
        guess = "content" if "content" in cols else cols[0]
        text_col = st.selectbox("Kolom teks", cols, index=cols.index(guess))
        st.session_state.text_col = text_col

        st.markdown("<p class='btn-hint'>Setelah kolom dipilih, lanjut ke menu <b>Preprocessing</b>.</p>", unsafe_allow_html=True)

        col_next, _ = st.columns([1, 4])
        with col_next:
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
    bright_header(
        "🧼 Preprocessing",
        "Klik tombol untuk menjalankan preprocessing. Hasil akan muncul bertahap agar mudah dibandingkan.",
    )

    if st.session_state.raw_df is None or not st.session_state.text_col:
        st.warning("Dataset atau kolom teks belum siap. Silakan ke menu Dataset dulu.")
        st.stop()

    ensure_nltk()

    # konfigurasi file kamus & lexicon
    ASSETS_DIR = "assets"
    kamus_path = os.path.join(ASSETS_DIR, "kamuskatabaku (1).xlsx")
    lex_pos_path = os.path.join(ASSETS_DIR, "positive.csv")
    lex_neg_path = os.path.join(ASSETS_DIR, "negative.csv")

    colL, colR = st.columns([1.1, 1])
    with colL:
        card_open()
        st.markdown("#### Konfigurasi file kamus & lexicon")
        st.caption("Pastikan file ada di folder assets/. Kalau nama file berbeda, sesuaikan di kode atau rename filenya.")
        st.write("📘 Kamus:", kamus_path)
        st.write("🟢 Lexicon positif:", lex_pos_path)
        st.write("🔴 Lexicon negatif:", lex_neg_path)

        missing = []
        for p in [kamus_path, lex_pos_path, lex_neg_path]:
            if not os.path.exists(p):
                missing.append(p)

        if missing:
            st.error("File berikut tidak ditemukan:")
            for m in missing:
                st.code(m)
            st.stop()
        card_close()

    with colR:
        card_open()
        st.markdown("#### Opsi output")
        drop_neutral = st.checkbox("Hapus data netral (score = 0)", value=True)
        st.caption("Ini mengikuti bagian akhir kode kamu (hapus netral).")
        card_close()

    st.markdown("")
    card_open()
    st.markdown("#### Jalankan preprocessing")
    run_prep = st.button("⚙️ Proses Preprocessing Sekarang", use_container_width=True)
    card_close()

    # tampilkan perbandingan step-by-step
    def show_compare(step_name, before_df, after_df, text_col="content", preview_rows=15):
        st.markdown("")
        card_open()
        st.markdown(f"### {step_name}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sebelum**")
            st.dataframe(before_df[[text_col]].head(preview_rows), use_container_width=True)
        with c2:
            st.markdown("**Sesudah**")
            st.dataframe(after_df[[text_col]].head(preview_rows), use_container_width=True)
        card_close()

    if run_prep:
        # base df
        base = st.session_state.raw_df.copy()
        text_col = st.session_state.text_col

        # buat kolom 'content' sebagai kolom kerja agar preprocessing mengikuti kode kamu
        df0 = base.copy()
        df0["content"] = df0[text_col].astype(str)

        st.session_state.prep_steps = {}
        st.session_state.prep_steps["0) Data Awal"] = df0.copy()

        # 1) Case Folding
        df1 = df0.copy()
        df1["content"] = df1["content"].apply(case_folding)
        st.session_state.prep_steps["1) Case Folding"] = df1.copy()

        # 2) Load Kamus + Normalisasi
        kamus_dict = load_kamus_excel(kamus_path)

        df2 = df1.copy()
        df2["content"] = df2["content"].apply(lambda x: normalisasi_dengan_kamus(x, kamus_dict))
        st.session_state.prep_steps["2) Normalisasi (Kamus)"] = df2.copy()

        # 3) Data Cleansing
        df3 = df2.copy()
        df3["content"] = df3["content"].apply(data_cleansing)
        st.session_state.prep_steps["3) Data Cleansing"] = df3.copy()

        # 4) Stopword Removal
        df4 = df3.copy()
        df4["content"] = df4["content"].apply(remove_stopwords_id)
        st.session_state.prep_steps["4) Stopword Removal"] = df4.copy()

        # 5) Stemming
        df5 = df4.copy()
        df5["content"] = df5["content"].apply(stem_text)
        st.session_state.prep_steps["5) Stemming"] = df5.copy()

        # 6) Tokenizing
        df6 = df5.copy()
        df6["content_list"] = df6["content"].apply(lambda x: word_tokenize(str(x)) if str(x).strip() else [])
        st.session_state.prep_steps["6) Tokenizing"] = df6.copy()

        # 7) Load lexicon + Filter token + Scoring + Sentimen
        lex_pos = load_lexicon(lex_pos_path)
        lex_neg = load_lexicon(lex_neg_path)

        df7 = df6.copy()
        df7["content_list"] = df7["content_list"].apply(lambda toks: filter_tokens_by_lexicon(toks, lex_pos, lex_neg))
        df7["content"] = df7["content_list"].apply(lambda toks: " ".join(toks))

        # hapus baris kosong (opsional, saya aktifkan karena sesuai praktek umum & mirip kode kamu)
        df7 = df7[df7["content_list"].map(len) > 0].reset_index(drop=True)

        res = df7["content_list"].apply(lambda toks: sentiment_analysis_lexicon_indonesia(toks, lex_pos, lex_neg))
        df7["score"] = res.apply(lambda x: x[0])
        df7["Sentimen"] = res.apply(lambda x: x[1])

        # Hapus netral
        if drop_neutral:
            df7 = df7[df7["score"] != 0].reset_index(drop=True)

        st.session_state.prep_steps["7) Pelabelan (Lexicon)"] = df7.copy()

        st.session_state.final_df = df7.copy()
        st.success("Preprocessing selesai. Scroll ke bawah untuk melihat perbandingan tiap tahap.")

    # render hasil step-by-step (kalau sudah ada)
    if st.session_state.prep_steps:
        keys = list(st.session_state.prep_steps.keys())

        # tampilkan ringkasan step
        st.markdown("")
        card_open()
        st.markdown("#### Ringkasan langkah yang tersedia")
        for k in keys:
            st.markdown(f"- {k}")
        card_close()

        # tampilkan perbandingan berurutan (kecuali step 0)
        for i in range(1, len(keys)):
            before_name = keys[i - 1]
            after_name = keys[i]
            before_df = st.session_state.prep_steps[before_name]
            after_df = st.session_state.prep_steps[after_name]

            if after_name.startswith("6) Tokenizing"):
                # tokenizing: compare content + content_list
                st.markdown("")
                card_open()
                st.markdown("### 6) Tokenizing")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Sebelum (teks)**")
                    st.dataframe(before_df[["content"]].head(15), use_container_width=True)
                with c2:
                    st.markdown("**Sesudah (list token)**")
                    st.dataframe(after_df[["content", "content_list"]].head(15), use_container_width=True)
                card_close()
            elif after_name.startswith("7) Pelabelan"):
                st.markdown("")
                card_open()
                st.markdown("### 7) Pelabelan (Lexicon)")
                st.dataframe(after_df[["content", "content_list", "score", "Sentimen"]].head(30), use_container_width=True)
                st.caption("Kolom content sudah difilter agar hanya menyisakan kata yang ada di lexicon (+/-), mengikuti acuan kode kamu.")
                card_close()
            else:
                show_compare(after_name, before_df, after_df, text_col="content")

        st.markdown("")
        col_go, _ = st.columns([1, 4])
        with col_go:
            if st.button("➡️ Lanjut ke Klasifikasi SVM", use_container_width=True):
                st.session_state.menu = "Klasifikasi SVM"
                st.rerun()


# =========================
# MENU: KLASIFIKASI SVM
# =========================
elif st.session_state.menu == "Klasifikasi SVM":
    bright_header(
        "🧠 Klasifikasi SVM",
        "Klik tombol untuk menjalankan SVM (TF-IDF, split 80/20). Fokus pada hasil: confusion matrix, report, akurasi.",
    )

    if st.session_state.final_df is None:
        st.warning("Data hasil preprocessing belum ada. Jalankan menu Preprocessing dulu.")
        st.stop()

    df = st.session_state.final_df.copy()

    # Validasi minimal
    needed_cols = {"content", "Sentimen"}
    if not needed_cols.issubset(df.columns):
        st.error(f"Kolom wajib tidak lengkap. Harus ada: {needed_cols}")
        st.stop()

    # UI sederhana
    card_open()
    st.markdown("#### Jalankan klasifikasi")
    st.caption("Catatan: proses TF-IDF dan split tidak ditampilkan (sesuai permintaan).")

    run_svm = st.button("🚀 Mulai Klasifikasi SVM", use_container_width=True)
    card_close()

    if run_svm:
        X = df["content"].astype(str).fillna("")
        y = df["Sentimen"].astype(str)

        # Split 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(y.unique()) > 1 else None
        )

        # TF-IDF
        tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2)
        )
        X_train_vec = tfidf.fit_transform(X_train)
        X_test_vec = tfidf.transform(X_test)

        # SVM (LinearSVC untuk teks umumnya cepat & kuat)
        model = LinearSVC()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        labels = sorted(y.unique())

        # Confusion Matrix plot
        st.markdown("")
        card_open()
        st.markdown("### ✅ Hasil Klasifikasi")
        st.markdown(f"**Akurasi:** `{acc:.4f}`")
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

        #

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Analisis Sentimen MBG",
    page_icon="🎯",
    layout="wide"
)

# ============================================================
# LOAD NAIVE BAYES MODEL & COMPONENTS
# ============================================================
@st.cache_resource
def load_naive_bayes_models():
    model = pickle.load(open('models/model.pkl', 'rb'))
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
    return model, vectorizer, label_encoder

nb_model, vectorizer, label_encoder = load_naive_bayes_models()

# ============================================================
# LOAD INDOBERT MODEL
# ============================================================
@st.cache_resource
def load_indobert_model():
    try:
        from transformers import pipeline, BertForSequenceClassification, BertTokenizer
        import os
        
        local_model_path = "./models/indobert"
        
        # Cek apakah ada model fine-tuned lokal
        if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "config.json")):
            print("📂 Loading fine-tuned IndoBERT from local...")
            model = BertForSequenceClassification.from_pretrained(local_model_path)
            tokenizer = BertTokenizer.from_pretrained(local_model_path)
            classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            return classifier, True, "fine-tuned"
        else:
            # Pakai pre-trained dari HuggingFace
            print("🌐 Loading pre-trained IndoBERT from HuggingFace...")
            classifier = pipeline(
                "sentiment-analysis",
                model="mdhugol/indonesia-bert-sentiment-classification"
            )
            return classifier, True, "pre-trained"
    except Exception as e:
        print(f"❌ Error loading IndoBERT: {e}")
        return None, False, None

indobert_classifier, indobert_available, indobert_type = load_indobert_model()

# ============================================================
# LEXICON KATA SENTIMEN (dari analisis_sentimen.py)
# ============================================================
positive_lexicon = {
    'bagus','baik','mantap','setuju','bermanfaat','membantu','rapi',
    'berhasil','meningkat','positif','patut','nyata','hebat','tepat',
    'inisiatif','layak','keren','oke','optimal'
}

negative_lexicon = {
    'racun','korupsi','korup','gagal','buruk','jelek','parah','busuk',
    'bobrok','bohong','rusak','bahaya','keracunan','kecewa','sampah',
    'ancam','merugikan','jahat','bobol','celaka'
}

# ============================================================
# PREPROCESSING FUNCTION (sama dengan di training)
# ============================================================
# Download stopwords jika belum ada
try:
    stop_words_id = set(stopwords.words('indonesian'))
except:
    nltk.download('stopwords')
    stop_words_id = set(stopwords.words('indonesian'))

def clean_youtube_comment(text):
    """Fungsi cleaning yang sama dengan di analisis_sentimen.py"""
    if not isinstance(text, str):
        return "", []
    
    original_text = text
    steps = []
    
    # 1. Ubah ke huruf kecil
    text = text.lower()
    steps.append(("Lowercase", text))
    
    # 2. Hapus URL, mention, dan hashtag
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    steps.append(("Hapus URL/Mention/Hashtag", text))
    
    # 3. Hapus karakter YouTube khusus
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    steps.append(("Hapus karakter khusus", text))
    
    # 4. Hapus karakter berulang (>2)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    steps.append(("Hapus karakter berulang", text))
    
    # 5. Hapus angka dan tanda baca
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    steps.append(("Hapus angka & tanda baca", text))
    
    # 6. Normalisasi spasi
    text = re.sub(r'\s+', ' ', text).strip()
    steps.append(("Normalisasi spasi", text))
    
    # 7. Hapus stopwords tapi pertahankan kata negasi penting
    important_words = {'tidak', 'bukan', 'jangan', 'belum', 'tak', 'ga', 'gak', 'nggak'}
    sentiment_whitelist = {'keren','mantap','layak','berhasil', 'bagus','baik','hebat','tepat', 'puas','bermanfaat','membantu', 'sekali','banget','sangat'}
    stop_words_filtered = stop_words_id - important_words
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words_filtered or word in sentiment_whitelist]
    text = ' '.join(filtered_words)
    steps.append(("Hapus stopwords", text))
    
    return text, steps

def find_sentiment_words(text, lexicon):
    """Cari kata-kata sentiment dalam teks"""
    words = text.lower().split()
    found = [word for word in words if word in lexicon]
    return found

# ============================================================
# HEADER
# ============================================================
st.title("🎯 Analisis Sentimen MBG")
st.markdown("**Makan Bergizi Gratis** - Analisis sentimen komentar")

# ============================================================
# MODEL SELECTOR (Dropdown sebelum tabs)
# ============================================================
st.markdown("---")

col_model, col_info = st.columns([2, 3])

with col_model:
    model_options = ["Naive Bayes + TF-IDF"]
    if indobert_available:
        model_options.append("IndoBERT (Pre-trained)")
    else:
        model_options.append("IndoBERT (Tidak tersedia)")
    
    selected_model = st.selectbox(
        "🤖 Pilih Model:",
        model_options,
        index=0,
        help="Pilih model yang akan digunakan untuk analisis sentimen"
    )



st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Analisis", "🔧 Pre-processing", "📝 Word Counter"])

# Session state untuk menyimpan hasil analisis
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'cleaned_text' not in st.session_state:
    st.session_state.cleaned_text = None
if 'cleaning_steps' not in st.session_state:
    st.session_state.cleaning_steps = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None
if 'original_text' not in st.session_state:
    st.session_state.original_text = None
if 'model_used' not in st.session_state:
    st.session_state.model_used = None

# Emoji dan color maps
emoji_map = {
    'positif': '😊',
    'positive': '😊',
    'POSITIVE': '😊',
    'LABEL_2': '😊',
    'negatif': '😢',
    'negative': '😢',
    'NEGATIVE': '😢',
    'LABEL_0': '😢',
    'netral': '😐',
    'neutral': '😐',
    'NEUTRAL': '😐',
    'LABEL_1': '😐'
}

color_map = {
    'positif': 'green',
    'positive': 'green',
    'POSITIVE': 'green',
    'LABEL_2': 'green',
    'negatif': 'red',
    'negative': 'red',
    'NEGATIVE': 'red',
    'LABEL_0': 'red',
    'netral': 'orange',
    'neutral': 'orange',
    'NEUTRAL': 'orange',
    'LABEL_1': 'orange'
}

# Label mapping untuk IndoBERT
indobert_label_map = {
    'LABEL_0': 'negatif',
    'LABEL_1': 'netral',
    'LABEL_2': 'positif'
}

# ============================================================
# TAB 1: ANALISIS
# ============================================================
with tab1:
    st.header("📊 Analisis Sentimen")
    
    # Tampilkan model yang dipilih
    st.caption(f"Model aktif: **{selected_model}**")
    
    # Input text
    user_input = st.text_area(
        "Masukkan komentar untuk dianalisis:",
        placeholder="Contoh: Program MBG sangat membantu anak-anak sekolah...",
        height=100
    )
    
    # Tombol analisis
    if st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True):
        if user_input.strip():
            # Proses cleaning (untuk kedua model)
            cleaned_text, cleaning_steps = clean_youtube_comment(user_input)
            
            if "Naive Bayes" in selected_model:
                # ===== NAIVE BAYES =====
                # Vectorize
                vector = vectorizer.transform([cleaned_text])
                
                # Predict
                prediction = nb_model.predict(vector)[0]
                probabilities = nb_model.predict_proba(vector)[0]
                
                # Decode label
                label = label_encoder.inverse_transform([prediction])[0]
                
                # Simpan ke session state
                st.session_state.original_text = user_input
                st.session_state.cleaned_text = cleaned_text
                st.session_state.cleaning_steps = cleaning_steps
                st.session_state.probabilities = probabilities
                st.session_state.model_used = "naive_bayes"
                st.session_state.analysis_result = {
                    'label': label,
                    'confidence': max(probabilities) * 100,
                    'prediction': prediction
                }
                
            elif "IndoBERT" in selected_model and indobert_available:
                # ===== INDOBERT =====
                with st.spinner("🔄 Memproses dengan IndoBERT..."):
                    result = indobert_classifier(user_input)[0]
                    
                    raw_label = result['label']
                    label = indobert_label_map.get(raw_label, raw_label.lower())
                    confidence = result['score'] * 100
                    
                    # Simpan ke session state
                    st.session_state.original_text = user_input
                    st.session_state.cleaned_text = cleaned_text
                    st.session_state.cleaning_steps = cleaning_steps
                    st.session_state.probabilities = None  # IndoBERT hanya kasih 1 skor
                    st.session_state.model_used = "indobert"
                    st.session_state.analysis_result = {
                        'label': label,
                        'confidence': confidence,
                        'prediction': raw_label
                    }
            else:
                st.error("❌ IndoBERT tidak tersedia. Silakan install transformers dan torch.")
        else:
            st.warning("⚠️ Mohon masukkan teks terlebih dahulu!")
    
    # Tampilkan hasil
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        st.markdown("---")
        st.subheader("✅ Hasil Analisis")
        
        # Show model used
        if st.session_state.model_used == "naive_bayes":
            st.caption("Diprediksi dengan: **Naive Bayes + TF-IDF**")
        else:
            st.caption("Diprediksi dengan: **IndoBERT**")
        
        emoji = emoji_map.get(result['label'], '🤔')
        color = color_map.get(result['label'], 'gray')
        
        # Layout hasil
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                <h1 style="font-size: 60px; margin: 0;">{emoji}</h1>
                <h2 style="color: {color}; margin: 10px 0;">{result['label'].upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h4>Confidence Score</h4>
                <h2 style="color: {color};">{result['confidence']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar untuk confidence
            st.progress(result['confidence'] / 100)
        
        # Tampilkan probabilitas semua kelas (hanya untuk Naive Bayes)
        if st.session_state.model_used == "naive_bayes" and st.session_state.probabilities is not None:
            st.markdown("##### Probabilitas per Kelas:")
            prob_df = pd.DataFrame({
                'Sentimen': label_encoder.classes_,
                'Probabilitas': [f"{p*100:.1f}%" for p in st.session_state.probabilities]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ============================================================
# TAB 2: PRE-PROCESSING
# ============================================================
with tab2:
    st.header("🔧 Proses Pre-processing")
    
    if st.session_state.original_text is None:
        st.info("💡 Lakukan analisis di tab 'Analisis' terlebih dahulu untuk melihat proses pre-processing.")
    else:
        # Model info
        if st.session_state.model_used == "naive_bayes":
            st.caption("Proses untuk: **Naive Bayes + TF-IDF**")
        else:
            st.caption("Proses untuk: **IndoBERT** (cleaning hanya untuk referensi)")
        
        # STEP 1: Text Cleaning
        st.subheader("📌 Step 1: Text Cleaning")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Teks Asli:**")
            st.code(st.session_state.original_text, language=None)
        with col2:
            st.markdown("**Teks Bersih:**")
            st.code(st.session_state.cleaned_text, language=None)
        
        # Detail langkah cleaning
        with st.expander("🔍 Lihat Detail Langkah Cleaning"):
            for i, (step_name, step_result) in enumerate(st.session_state.cleaning_steps, 1):
                st.markdown(f"**{i}. {step_name}:**")
                st.code(step_result if step_result else "(kosong)", language=None)
        
        st.markdown("---")
        
        if st.session_state.model_used == "naive_bayes":
            # STEP 2: TF-IDF Vectorization
            st.subheader("📌 Step 2: TF-IDF Vectorization")
            st.markdown("""
            Teks yang sudah bersih diubah menjadi vektor angka menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**.
            
            - **TF (Term Frequency)**: Seberapa sering kata muncul dalam dokumen
            - **IDF (Inverse Document Frequency)**: Seberapa unik kata tersebut di seluruh dataset
            - **Hasil**: Vektor dengan 6000 fitur (kata/bigram penting)
            """)
            
            # Show vector shape
            vector = vectorizer.transform([st.session_state.cleaned_text])
            st.code(f"Shape vektor: {vector.shape} (1 dokumen × 6000 fitur)", language=None)
            
            # Show non-zero features
            non_zero = vector.nnz
            st.code(f"Jumlah fitur non-zero: {non_zero} kata/bigram terdeteksi", language=None)
            
            st.markdown("---")
            
            # STEP 3: Model Prediction
            st.subheader("📌 Step 3: Model Prediction (Naive Bayes)")
            st.markdown("""
            Model **Multinomial Naive Bayes** menghitung probabilitas untuk setiap kelas sentimen:
            """)
            
            # Probabilitas
            if st.session_state.probabilities is not None:
                probs = st.session_state.probabilities
                classes = label_encoder.classes_
                
                for cls, prob in zip(classes, probs):
                    emoji = emoji_map.get(cls, '🤔')
                    st.markdown(f"- **{cls.upper()}** {emoji}: `{prob*100:.2f}%`")
                
                st.markdown(f"""
                **Prediksi:** Kelas dengan probabilitas tertinggi = **{classes[probs.argmax()].upper()}**
                """)
            
            st.markdown("---")
            
            # STEP 4: Label Decoding
            st.subheader("📌 Step 4: Label Decoding")
            st.markdown("""
            Hasil prediksi model (angka) diubah kembali menjadi label teks:
            """)
            
            result = st.session_state.analysis_result
            st.code(f"Prediksi Model: {result['prediction']} → Label: '{result['label']}'", language=None)
            
            st.markdown("""
            **Mapping Label Encoder:**
            - 0 → negatif
            - 1 → netral  
            - 2 → positif
            """)
        else:
            # IndoBERT processing
            st.subheader("📌 Step 2: Tokenization (IndoBERT)")
            st.markdown("""
            IndoBERT menggunakan **WordPiece Tokenization**:
            
            - Teks dipecah menjadi sub-kata (subwords)
            - Setiap token diubah menjadi ID numerik
            - Ditambahkan token khusus: [CLS] di awal, [SEP] di akhir
            """)
            
            st.markdown("---")
            
            st.subheader("📌 Step 3: BERT Encoding")
            st.markdown("""
            Model BERT memproses token:
            
            - 12 layer transformer
            - Attention mechanism untuk memahami konteks
            - Output: representasi vektor 768 dimensi
            """)
            
            st.markdown("---")
            
            st.subheader("📌 Step 4: Classification")
            st.markdown("""
            Layer klasifikasi di akhir model:
            
            - Mengubah representasi BERT → probabilitas kelas
            - 3 kelas: LABEL_0 (negatif), LABEL_1 (netral), LABEL_2 (positif)
            """)
            
            result = st.session_state.analysis_result
            st.code(f"Prediksi: {result['prediction']} → Label: '{result['label']}' (Confidence: {result['confidence']:.1f}%)", language=None)

# ============================================================
# TAB 3: WORD COUNTER
# ============================================================
with tab3:
    st.header("📝 Word Counter - Kata Sentimen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("😊 Kata Positif (Lexicon)")
        st.markdown("Kata-kata yang mengindikasikan sentimen **positif**:")
        
        # Tampilkan lexicon positif
        pos_words_display = ", ".join(sorted(positive_lexicon))
        st.info(pos_words_display)
    
    with col2:
        st.subheader("😢 Kata Negatif (Lexicon)")
        st.markdown("Kata-kata yang mengindikasikan sentimen **negatif**:")
        
        # Tampilkan lexicon negatif
        neg_words_display = ", ".join(sorted(negative_lexicon))
        st.error(neg_words_display)
    
    st.markdown("---")
    
    # Analisis kata sentiment dari input user
    st.subheader("🔍 Kata Sentimen dari Input Anda")
    
    if st.session_state.original_text is None:
        st.info("💡 Lakukan analisis di tab 'Analisis' terlebih dahulu untuk melihat kata sentimen dalam input Anda.")
    else:
        st.markdown(f"**Teks yang dianalisis:**")
        st.code(st.session_state.original_text, language=None)
        
        # Cari kata positif dan negatif dalam input
        cleaned = st.session_state.cleaned_text
        found_positive = find_sentiment_words(cleaned, positive_lexicon)
        found_negative = find_sentiment_words(cleaned, negative_lexicon)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ✅ Kata Positif Ditemukan:")
            if found_positive:
                for word in set(found_positive):
                    count = found_positive.count(word)
                    st.success(f"**{word}** ({count}x)")
            else:
                st.warning("Tidak ada kata positif ditemukan")
        
        with col2:
            st.markdown("##### ❌ Kata Negatif Ditemukan:")
            if found_negative:
                for word in set(found_negative):
                    count = found_negative.count(word)
                    st.error(f"**{word}** ({count}x)")
            else:
                st.warning("Tidak ada kata negatif ditemukan")
        
        # Summary
        st.markdown("---")
        st.markdown("##### 📊 Summary")
        
        total_pos = len(found_positive)
        total_neg = len(found_negative)
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Kata Positif", total_pos, delta=None)
        with summary_col2:
            st.metric("Kata Negatif", total_neg, delta=None)
        with summary_col3:
            if total_pos > total_neg:
                st.metric("Dominan", "Positif 😊")
            elif total_neg > total_pos:
                st.metric("Dominan", "Negatif 😢")
            else:
                st.metric("Dominan", "Netral 😐")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>📊 Naive Bayes: Akurasi 87% (3263 data MBG) | IndoBERT: Pre-trained model untuk Bahasa Indonesia</p>
</div>
""", unsafe_allow_html=True)

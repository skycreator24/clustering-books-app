import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from yellowbrick.cluster import KElbowVisualizer
import io
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Dashboard Clustering Perpustakaan", layout="wide")
st.title("📊 Aplikasi Clustering & Profiling Data Perpustakaan")
st.write(
    "Upload dataset peminjaman perpustakaan Anda, lakukan pra-pemrosesan otomatis, "
    "dan jalankan analisis K-Means dengan visualisasi PCA serta evaluasi Silhouette Score."
)

# ==========================================
# 1. FUNGSI PREPROCESSING DATA (CACHED)
# ==========================================
# Menggunakan cache agar Streamlit tidak mengulang proses berat ini setiap kali tombol diklik
@st.cache_data
def preprocess_data(df):
    df_clean = df.dropna().copy()

    # Split bibliografi
    if 'Data Bibliografis' in df_clean.columns:
        bibliografi_split = df_clean['Data Bibliografis'].astype(str).str.split('/', expand=True)
        df_clean['Judul'] = bibliografi_split[0].str.strip() if 0 in bibliografi_split.columns else ''
        df_clean['Pengarang'] = bibliografi_split[1].str.strip() if 1 in bibliografi_split.columns else ''

    # Split penerbit[cite: 2]
    if 'Penerbit' in df_clean.columns:
        penerbit_split = df_clean['Penerbit'].astype(str).str.split(',', expand=True, n=1)
        lokasi_nama = penerbit_split[0].str.split(':', expand=True, n=1)
        df_clean['Lokasi_Penerbit'] = lokasi_nama[0].str.strip() if 0 in lokasi_nama.columns else ''
        df_clean['Nama_Penerbit'] = lokasi_nama[1].str.strip() if 1 in lokasi_nama.columns else ''
        df_clean['Tahun_Penerbit'] = penerbit_split[1].str.strip() if 1 in penerbit_split.columns else ''

    # Normalize Text[cite: 2]
    def normalize_text(text):
        if isinstance(text, str):
            return text.lower().strip()
        return text

    for col in ['Lokasi_Penerbit', 'Nama_Penerbit', 'Pengarang']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(normalize_text)

    # Drop columns[cite: 2]
    columns_to_drop = ['Tanggal Pinjam', 'Tanggal Dikembalikan', 'Data Bibliografis', 'Penerbit', 'Nomor Klass']
    existing_cols_to_drop = [c for c in columns_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=existing_cols_to_drop)
    df_clean = df_clean.dropna()

    # Kategori Mapping[cite: 2]
    kategori_mapping = {
        'Ilmu Komputer': 'Teknologi Informasi', 'Teknologi Informasi': 'Teknologi Informasi',
        'Teknologi informasi': 'Teknologi Informasi', 'bacaan anak': 'Buku Anak',
        'Buku Anak': 'Buku Anak', 'Komik': 'Buku Anak', 'Fiksi': 'Fiksi & Sastra',
        'Novel': 'Fiksi & Sastra', 'Fabel': 'Fiksi & Sastra', 'Cerita Pendek': 'Fiksi & Sastra',
        'Sastra Islam': 'Fiksi & Sastra', 'Mitologi': 'Fiksi & Sastra', 'Ekonomi': 'Bisnis & Ekonomi',
        'Manajemen': 'Bisnis & Ekonomi', 'Perbankan': 'Bisnis & Ekonomi', 'Akuntansi': 'Bisnis & Ekonomi',
        'Bisnis dan Manajemen': 'Bisnis & Ekonomi', 'Investasi': 'Bisnis & Ekonomi',
        'Agama Nusantara': 'Agama', 'Agama Islam': 'Agama', 'Okultisme': 'Agama',
        'Sejarah': 'Sejarah & Biografi', 'Sejarah Islam': 'Sejarah & Biografi',
        'Sejarah Dunia': 'Sejarah & Biografi', 'Sejarah Indonesia': 'Sejarah & Biografi',
        'Biografi': 'Sejarah & Biografi', 'Self Improvement': 'Pengembangan Diri',
        'Psikologi': 'Pengembangan Diri', 'Filsafat': 'Pengembangan Diri', 'Motivasi': 'Pengembangan Diri',
        'Pendidikan': 'Pendidikan', 'Sains': 'Sains', 'Astronomi': 'Sains', 'Matematika': 'Sains',
        'Ilmu Pengetahuan Alam': 'Sains', 'Kesehatan': 'Ilmu Terapan', 'Ilmu Kesehatan': 'Ilmu Terapan',
        'Cinematography': 'Ilmu Terapan', 'Pertanian': 'Ilmu Terapan', 'Keterampilan': 'Ilmu Terapan',
        'Bahasa Inggris': 'Bahasa', 'Bahasa Arab': 'Bahasa', 'Bahasa Jepang': 'Bahasa',
        'Politik': 'Sosial & Politik', 'Wawasan Kebangsaan': 'Sosial & Politik',
        'Sosiologi': 'Sosial & Politik', 'Komunikasi': 'Sosial & Politik', 'Ilmu Komunikasi': 'Sosial & Politik',
        'Hukum': 'Sosial & Politik', 'Adat Istiadat': 'Sosial & Politik', 'Adat istiadat': 'Sosial & Politik',
        'Seni Rupa': 'Gaya Hidup & Hobi', 'Masakan': 'Gaya Hidup & Hobi', 'Musik': 'Gaya Hidup & Hobi',
        'Kuliner': 'Gaya Hidup & Hobi'
    }
    
    if 'Kategori Buku' in df_clean.columns:
        df_clean['kategori_buku_clean'] = df_clean['Kategori Buku'].replace(kategori_mapping)
        df_clean = df_clean.drop('Kategori Buku', axis=1)

    # Transformasi Numerik[cite: 2]
    if 'Jumlah Hari Telat' in df_clean.columns:
        df_clean['Jumlah Hari Telat'] = pd.to_numeric(df_clean['Jumlah Hari Telat'], errors='coerce').fillna(0)
        df_clean['Jumlah Hari Telat'] = (df_clean['Jumlah Hari Telat'] * -1) + 7
    
    if 'Tahun_Penerbit' in df_clean.columns:
        df_clean['Tahun_Penerbit'] = pd.to_numeric(df_clean['Tahun_Penerbit'], errors='coerce')
        df_clean['Tahun_Penerbit'] = df_clean['Tahun_Penerbit'].fillna(df_clean['Tahun_Penerbit'].median())
        df_clean['Umur_Buku'] = 2026 - df_clean['Tahun_Penerbit']

    return df_clean

# ==========================================
# 2. SIDEBAR (UPLOAD & PENGATURAN)
# ==========================================
st.sidebar.header("📁 1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file Excel/CSV", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Error Handling Pembacaan File
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # Memproses Data
    with st.spinner("Memproses dan membersihkan data..."):
        df_clean = preprocess_data(df_raw)
    
    # Pengaturan K-Means di Sidebar
    st.sidebar.header("⚙️ 2. Pengaturan K-Means")
    num_clusters = st.sidebar.slider("Pilih Jumlah Klaster (K) Final", min_value=2, max_value=10, value=5)

    # Membuat Layout Tab
    tab1, tab2, tab3, tab4 = st.tabs(["🗃️ Data Preview", "📈 Exploratory Data Analysis", "🤖 K-Means Clustering", "📋 Analisis & Download"])

    # ==========================================
    # TAB 1: DATA PREVIEW
    # ==========================================
    with tab1:
        st.subheader("Perbandingan Data")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Data Asli ({df_raw.shape[0]} Baris)**")
            st.dataframe(df_raw.head(), use_container_width=True)
        with col2:
            st.markdown(f"**Data Bersih ({df_clean.shape[0]} Baris)**")
            st.dataframe(df_clean.head(), use_container_width=True)
            
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.write("**Missing Values Tersisa:**", df_clean.isnull().sum())
        with col4:
            st.write("**Unique Values (Kardinalitas):**", df_clean.nunique())

    # ==========================================
    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================
    with tab2:
        st.subheader("Eksplorasi Karakteristik Buku")
        
        # Wordcloud[cite: 2]
        if 'Judul' in df_clean.columns:
            st.markdown("#### Topik Terpopuler (WordCloud Judul)")
            text = " ".join(df_clean["Judul"].astype(str))
            wc = WordCloud(width=1200, height=400, background_color='white').generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(12, 4))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

        col1, col2 = st.columns(2)
        with col1:
            if 'Umur_Buku' in df_clean.columns:
                st.markdown("#### Distribusi Umur Buku")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.histplot(df_clean["Umur_Buku"], bins=20, kde=True, ax=ax1, color='skyblue')
                st.pyplot(fig1)
            
        with col2:
            if 'kategori_buku_clean' in df_clean.columns:
                st.markdown("#### Distribusi Kategori Buku")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                kategori_counts = df_clean['kategori_buku_clean'].value_counts().reset_index()
                kategori_counts.columns = ['Kategori', 'Jumlah']
                sns.barplot(data=kategori_counts, x="Jumlah", y="Kategori", palette="viridis", ax=ax2)
                st.pyplot(fig2)

    # ==========================================
    # TAB 3: K-MEANS CLUSTERING
    # ==========================================
    with tab3:
        st.subheader("Proses Clustering (Auto-Encoding & PCA)")
        
        # Persiapan Fitur Otomatis[cite: 2]
        kandidat_fitur = ['Jumlah Hari Telat', 'Umur_Buku', 'kategori_buku_clean', 'Lokasi_Penerbit', 'Nama_Penerbit']
        fitur_tersedia = [f for f in kandidat_fitur if f in df_clean.columns]
        
        df_prep = df_clean[fitur_tersedia].copy()
        
        kolom_numerik = [c for c in ['Jumlah Hari Telat', 'Umur_Buku'] if c in df_prep.columns]
        
        if len(kolom_numerik) > 0:
            scaler = MinMaxScaler()
            df_prep[kolom_numerik] = scaler.fit_transform(df_prep[kolom_numerik])
            
        X_final = pd.get_dummies(df_prep)

        col1, col2 = st.columns(2)
        
        # Evaluasi Elbow Method[cite: 2]
        with col1:
            st.markdown("#### Pencarian K Optimal (Metode Elbow)")
            fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
            model_elbow = KMeans(random_state=42, n_init=10)
            model_elbow._estimator_type = "clusterer"
            visualizer = KElbowVisualizer(model_elbow, k=(2, 10), metric='distortion', ax=ax_elbow)
            visualizer.fit(X_final)
            visualizer.finalize()
            st.pyplot(fig_elbow)
            st.info(f"Saran dari sistem: K = {visualizer.elbow_value_}")

        # Eksekusi Model Berdasarkan Input Sidebar
        model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df_clean['Cluster_Final'] = model.fit_predict(X_final)
        score = silhouette_score(X_final, df_clean['Cluster_Final'])
        
        # PCA Scatter Plot[cite: 2]
        with col2:
            st.markdown(f"#### Persebaran Data (PCA Scatter Plot K={num_clusters})")
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_final)
            centroids_pca = pca.transform(model.cluster_centers_)

            fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_clean['Cluster_Final'], palette='Set1', s=100, alpha=0.7, edgecolor='w', ax=ax_pca)
            ax_pca.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, color='black', label='Centroids', zorder=10)
            ax_pca.set_title(f'Silhouette Score: {score:.4f}')
            ax_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig_pca)

    # ==========================================
    # TAB 4: ANALISIS KLASTER & DOWNLOAD
    # ==========================================
    with tab4:
        st.subheader("Profil & Karakteristik Klaster")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Statistik Numerik per Klaster")
            if len(kolom_numerik) > 0:
                statistik_lengkap = df_clean.groupby('Cluster_Final')[kolom_numerik].agg(['mean', 'min', 'max']).round(2)
                st.dataframe(statistik_lengkap, use_container_width=True)
                
        with col2:
            st.markdown("#### Kategori Dominan")
            if 'kategori_buku_clean' in df_clean.columns:
                profile_pct = pd.crosstab(df_clean['Cluster_Final'], df_clean['kategori_buku_clean'], normalize='index') * 100
                dominant_category = profile_pct.idxmax(axis=1)
                st.dataframe(dominant_category.rename("Kategori Dominan"), use_container_width=True)

        if 'kategori_buku_clean' in df_clean.columns:
            st.markdown("#### Distribusi Kategori Buku per Klaster")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            profile_pct.plot(kind='bar', stacked=True, colormap='tab20', ax=ax_bar)
            ax_bar.legend(title='Kategori Buku', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig_bar)

        st.markdown("---")
        st.subheader("Simpan Hasil Analisis")
        
        # Fitur Download CSV dari code lama[cite: 3]
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Dataset + Hasil Clustering (CSV)",
            data=csv_buffer.getvalue(),
            file_name="hasil_clustering_perpustakaan.csv",
            mime="text/csv",
        )
        st.markdown("---")
        st.markdown("#### Hubungan Umur Buku & Hari Telat per Klaster")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
        
        # Cek apakah kolomnya ada sebelum menggambar
        if 'Umur_Buku' in df_clean.columns and 'Jumlah Hari Telat' in df_clean.columns:
            sns.scatterplot(
                data=df_clean,
                x="Umur_Buku",
                y="Jumlah Hari Telat",
                hue="Cluster_Final",
                palette="Set1",
                s=100,
                alpha=0.7,
                ax=ax_scatter
            )
            ax_scatter.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig_scatter)

        st.markdown("#### Distribusi Hari Telat per Klaster")
        fig_box2, ax_box2 = plt.subplots(figsize=(8, 5))
        if 'Jumlah Hari Telat' in df_clean.columns:
            sns.boxplot(data=df_clean, x="Cluster_Final", y="Jumlah Hari Telat", palette="Set2", ax=ax_box2)
            st.pyplot(fig_box2)

        st.markdown("---")
        st.markdown("#### Heatmap Korelasi Variabel Numerik")
        kolom_num = [c for c in ['Jumlah Hari Telat', 'Umur_Buku'] if c in df_clean.columns]
        
        if len(kolom_num) > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            corr = df_clean[kolom_num].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)

else:
    # Tampilan Awal Sebelum Upload[cite: 3]
    st.info("👈 Silakan upload file Excel atau CSV perpustakaan Anda di sidebar sebelah kiri untuk memulai analisis.")
    st.image("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True)

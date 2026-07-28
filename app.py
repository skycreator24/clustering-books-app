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

# Konfigurasi Halaman Streamlit[cite: 4]
st.set_page_config(page_title="Dashboard Clustering Perpustakaan", layout="wide")
st.title("📊 Aplikasi Clustering & Profiling Data Perpustakaan")
st.write(
    "Upload dataset peminjaman perpustakaan Anda, pilih fitur yang ingin dianalisis, "
    "dan biarkan sistem mencari jumlah klaster (K) paling optimal secara otomatis."
)

# ==========================================
# 1. FUNGSI PREPROCESSING DATA (CACHED)
# ==========================================
@st.cache_data
def preprocess_data(df):
    df_clean = df.dropna().copy()

    # Split bibliografi[cite: 4]
    if 'Data Bibliografis' in df_clean.columns:
        bibliografi_split = df_clean['Data Bibliografis'].astype(str).str.split('/', expand=True)
        df_clean['Judul'] = bibliografi_split[0].str.strip() if 0 in bibliografi_split.columns else ''
        df_clean['Pengarang'] = bibliografi_split[1].str.strip() if 1 in bibliografi_split.columns else ''

    # Split penerbit[cite: 4]
    if 'Penerbit' in df_clean.columns:
        penerbit_split = df_clean['Penerbit'].astype(str).str.split(',', expand=True, n=1)
        lokasi_nama = penerbit_split[0].str.split(':', expand=True, n=1)
        df_clean['Lokasi_Penerbit'] = lokasi_nama[0].str.strip() if 0 in lokasi_nama.columns else ''
        df_clean['Nama_Penerbit'] = lokasi_nama[1].str.strip() if 1 in lokasi_nama.columns else ''
        df_clean['Tahun_Penerbit'] = penerbit_split[1].str.strip() if 1 in penerbit_split.columns else ''

    # Normalize Text[cite: 4]
    def normalize_text(text):
        if isinstance(text, str):
            return text.lower().strip()
        return text

    for col in ['Lokasi_Penerbit', 'Nama_Penerbit', 'Pengarang']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(normalize_text)

    # Drop columns[cite: 4]
    columns_to_drop = ['Tanggal Pinjam', 'Tanggal Dikembalikan', 'Data Bibliografis', 'Penerbit', 'Nomor Klass']
    existing_cols_to_drop = [c for c in columns_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=existing_cols_to_drop)
    df_clean = df_clean.dropna()

    # Kategori Mapping[cite: 4]
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

    # Transformasi Numerik[cite: 4]
    if 'Jumlah Hari Telat' in df_clean.columns:
        df_clean['Jumlah Hari Telat'] = pd.to_numeric(df_clean['Jumlah Hari Telat'], errors='coerce').fillna(0)
        df_clean['Jumlah Hari Telat'] = (df_clean['Jumlah Hari Telat'] * -1) + 7
    
    if 'Tahun_Penerbit' in df_clean.columns:
        df_clean['Tahun_Penerbit'] = pd.to_numeric(df_clean['Tahun_Penerbit'], errors='coerce')
        df_clean['Tahun_Penerbit'] = df_clean['Tahun_Penerbit'].fillna(df_clean['Tahun_Penerbit'].median())
        df_clean['Umur_Buku'] = 2026 - df_clean['Tahun_Penerbit']

    return df_clean

# ==========================================
# 2. SIDEBAR (UPLOAD & PEMILIHAN FITUR)
# ==========================================
st.sidebar.header("📁 1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file Excel/CSV", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Error Handling Pembacaan File[cite: 4]
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
    
    # ==========================================
    # PEMILIHAN FITUR DINAMIS DI SIDEBAR
    # ==========================================
    st.sidebar.header("⚙️ 2. Pilih Fitur Clustering")
    all_columns = df_clean.columns.tolist()
    
    # Menentukan nilai default jika kolom tersedia
    num_default = [c for c in ['Jumlah Hari Telat', 'Umur_Buku'] if c in all_columns]
    cat_default = [c for c in ['kategori_buku_clean', 'Lokasi_Penerbit', 'Nama_Penerbit'] if c in all_columns]

    selected_numeric = st.sidebar.multiselect(
        "Fitur Numerik (Skala 0-1):", 
        options=all_columns, 
        default=num_default
    )
    selected_categorical = st.sidebar.multiselect(
        "Fitur Kategorikal (One-Hot Encoding):", 
        options=[c for c in all_columns if c not in selected_numeric], 
        default=cat_default
    )

    if not selected_numeric and not selected_categorical:
        st.warning("⚠️ Silakan pilih minimal satu fitur numerik atau kategorikal di sidebar untuk memulai proses K-Means.")
        st.stop()

    # Membuat Layout Tab[cite: 4]
    tab1, tab2, tab3, tab4 = st.tabs(["🗃️ Data Preview", "📈 Exploratory Data Analysis", "🤖 Auto K-Means Clustering", "📋 Analisis & Download"])

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
        
        # Wordcloud[cite: 4]
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
                sns.barplot(data=kategori_counts.head(15), x="Jumlah", y="Kategori", palette="viridis", ax=ax2)
                st.pyplot(fig2)

    # ==========================================
    # TAB 3: AUTO K-MEANS CLUSTERING
    # ==========================================
    with tab3:
        st.subheader("Automasi Pencarian Klaster & Persebaran Data")
        
        # Penyatuan & Encoding Fitur yang Dipilih 
        X_parts = []
        if selected_numeric:
            scaler = MinMaxScaler()
            X_num = pd.DataFrame(scaler.fit_transform(df_clean[selected_numeric]), columns=selected_numeric, index=df_clean.index)
            X_parts.append(X_num)
        
        if selected_categorical:
            X_cat = pd.get_dummies(df_clean[selected_categorical].astype(str))
            X_parts.append(X_cat)
            
        X_final = pd.concat(X_parts, axis=1)

        col1, col2 = st.columns(2)
        
        # Evaluasi Elbow Method Otomatis[cite: 4]
        with col1:
            st.markdown("#### Pencarian K Optimal (Metode Elbow)")
            fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
            model_elbow = KMeans(random_state=42, n_init=10)
            model_elbow._estimator_type = "clusterer"
            
            # Batas uji K disesuaikan dengan ketersediaan data
            max_k = min(10, len(X_final) - 1)
            visualizer = KElbowVisualizer(model_elbow, k=(2, max_k), metric='distortion', ax=ax_elbow)
            
            with st.spinner("Mencari titik Elbow..."):
                visualizer.fit(X_final)
                visualizer.finalize()
            
            st.pyplot(fig_elbow)
            
            # Ekstrak nilai K terbaik dari Yellowbrick
            best_k = visualizer.elbow_value_
            
            if best_k is None:
                best_k = 2  # Fallback pengamanan
                st.warning("Peringatan: Titik potong Elbow tidak terdeteksi dengan tajam. Menggunakan default K=2.")
            else:
                st.success(f"🎯 Sistem secara otomatis menetapkan **K = {best_k}** berdasarkan analisis KElbowVisualizer.")

        # Eksekusi Model Final dengan 'best_k'[cite: 4]
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df_clean['Cluster_Final'] = model.fit_predict(X_final)
        score = silhouette_score(X_final, df_clean['Cluster_Final'])
        
        # PCA Scatter Plot
        with col2:
            st.markdown(f"#### Persebaran Data (PCA Scatter Plot K={best_k})")
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
            if len(selected_numeric) > 0:
                statistik_lengkap = df_clean.groupby('Cluster_Final')[selected_numeric].agg(['mean', 'min', 'max']).round(2)
                st.dataframe(statistik_lengkap, use_container_width=True)
            else:
                st.info("Tidak ada fitur numerik yang dipilih untuk dihitung statistiknya.")
                
        with col2:
            if len(selected_categorical) > 0:
                # Menggunakan fitur kategorikal utama (yang pertama dipilih) untuk analisis dominan
                profil_cat = selected_categorical[0] 
                st.markdown(f"#### Atribut Dominan ({profil_cat})")
                profile_pct = pd.crosstab(df_clean['Cluster_Final'], df_clean[profil_cat], normalize='index') * 100
                dominant_category = profile_pct.idxmax(axis=1)
                st.dataframe(dominant_category.rename(f"Mayoritas {profil_cat}"), use_container_width=True)

        if len(selected_categorical) > 0:
            profil_cat = selected_categorical[0]
            st.markdown(f"#### Distribusi {profil_cat} per Klaster")
            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            profile_pct = pd.crosstab(df_clean['Cluster_Final'], df_clean[profil_cat], normalize='index') * 100
            profile_pct.plot(kind='bar', stacked=True, colormap='tab20', ax=ax_bar)
            ax_bar.legend(title=profil_cat, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig_bar)
            
        st.markdown("---")
        st.markdown("#### Hubungan Umur Buku & Hari Telat per Klaster")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 5))
        
        # Grafik pelengkap akan muncul jika kolomnya memang ada[cite: 4]
        if 'Umur_Buku' in df_clean.columns and 'Jumlah Hari Telat' in df_clean.columns:
            sns.scatterplot(
                data=df_clean, x="Umur_Buku", y="Jumlah Hari Telat", hue="Cluster_Final",
                palette="Set1", s=100, alpha=0.7, ax=ax_scatter
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

        st.markdown("---")
        st.subheader("Simpan Hasil Analisis")
        
        # Fitur Download CSV[cite: 4]
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Dataset + Hasil Clustering (CSV)",
            data=csv_buffer.getvalue(),
            file_name="hasil_clustering_perpustakaan.csv",
            mime="text/csv",
        )

else:
    # Tampilan Awal Sebelum Upload[cite: 4]
    st.info("👈 Silakan upload file Excel atau CSV perpustakaan Anda di sidebar sebelah kiri untuk memulai analisis.")
    st.image("https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_column_width=True)

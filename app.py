import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import io

st.set_page_config(page_title="Clustering & Profiling App", layout="wide")

st.title("📊 Aplikasi Clustering & Profiling Data (K-Means + Silhouette Score)")
st.write(
    "Upload dataset kamu (CSV/Excel), pilih kolom yang ingin dipakai, lalu jalankan "
    "clustering otomatis dan lihat profiling tiap cluster. Aplikasi ini bersifat "
    "generik, jadi bisa dipakai untuk dataset apa saja — tidak hanya data buku perpustakaan."
)

uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    st.subheader("Preview Data")
    st.dataframe(df.head())
    st.write(f"Jumlah baris: **{df.shape[0]}**, Jumlah kolom: **{df.shape[1]}**")

    if st.checkbox("Hapus baris dengan nilai kosong (NaN)?", value=True):
        before = df.shape[0]
        df = df.dropna()
        st.write(f"Baris sebelum: {before} → setelah dibersihkan: {df.shape[0]}")

    all_columns = df.columns.tolist()
    numeric_default = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_all = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    st.subheader("💰 Alokasi Anggaran per Kategori Buku")
    st.write(
        "Masukkan total anggaran yang tersedia, pilih kolom kategori, dan anggaran akan "
        "dibagi otomatis sesuai persentase jumlah buku di tiap kategori."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        kategori_col = st.selectbox(
            "Kolom Kategori Buku:",
            options=cat_cols_all if cat_cols_all else all_columns,
        )
    with col_b:
        total_anggaran = st.number_input("Total Anggaran (Rp):", min_value=0, value=10_000_000, step=100_000)

    if st.button("Hitung Alokasi Anggaran"):
        if total_anggaran <= 0:
            st.warning("Masukkan jumlah anggaran lebih dari 0.")
        else:
            counts = df[kategori_col].value_counts()
            pct = (counts / counts.sum() * 100).round(2)
            alokasi = (pct / 100 * total_anggaran).round(0)

            def format_rupiah(x):
                return "Rp " + f"{x:,.0f}".replace(",", ".")

            hasil_anggaran = pd.DataFrame({
                "Kategori": counts.index,
                "Jumlah Buku": counts.values,
                "Persentase (%)": pct.values,
                "Alokasi Anggaran": [format_rupiah(a) for a in alokasi.values],
            })
            st.dataframe(hasil_anggaran, use_container_width=True)

            fig_pie, ax_pie = plt.subplots(figsize=(6.5, 6.5))
            ax_pie.pie(
                pct.values,
                labels=[f"{k}\n({p:.1f}%)" for k, p in zip(counts.index, pct.values)],
                autopct=lambda p: format_rupiah(p / 100 * total_anggaran),
                colors=plt.cm.tab20.colors,
                startangle=90,
                textprops={"fontsize": 8},
            )
            ax_pie.set_title(f"Alokasi Anggaran per {kategori_col}\nTotal: {format_rupiah(total_anggaran)}")
            plt.tight_layout()
            st.pyplot(fig_pie)

    st.subheader("1. Pilih Kolom untuk Clustering")
    col1, col2 = st.columns(2)
    with col1:
        numeric_cols = st.multiselect(
            "Kolom numerik (dinormalisasi dengan MinMaxScaler):",
            options=all_columns,
            default=numeric_default,
        )
    with col2:
        categorical_cols = st.multiselect(
            "Kolom kategorikal (di-encode dengan One-Hot Encoding):",
            options=[c for c in all_columns if c not in numeric_cols],
        )

    if not numeric_cols and not categorical_cols:
        st.warning("Pilih minimal satu kolom numerik atau kategorikal untuk melanjutkan.")
        st.stop()

    X_parts = []
    if numeric_cols:
        scaler = MinMaxScaler()
        X_num = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index
        )
        X_parts.append(X_num)
    if categorical_cols:
        X_cat = pd.get_dummies(df[categorical_cols].astype(str))
        X_parts.append(X_cat)
    X = pd.concat(X_parts, axis=1)

    st.subheader("2. Evaluasi Jumlah Cluster Terbaik (Silhouette Score)")
    k_min, k_max = st.slider("Rentang jumlah cluster (K) untuk dievaluasi:", 2, 15, (2, 8))
    k_range = list(range(k_min, k_max + 1))

    if st.button("Jalankan Evaluasi K"):
        skor_silhouette = []
        with st.spinner("Menghitung silhouette score untuk tiap K..."):
            for k in k_range:
                model_sementara = KMeans(n_clusters=k, random_state=42, n_init=10)
                label_sementara = model_sementara.fit_predict(X)
                skor = silhouette_score(X, label_sementara)
                skor_silhouette.append(skor)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_range, skor_silhouette, marker="o", linestyle="-", color="teal", linewidth=2)
        ax.set_title("Grafik Evaluasi K Terbaik")
        ax.set_xlabel("Jumlah Cluster (K)")
        ax.set_ylabel("Silhouette Score")
        ax.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig)

        hasil_k = pd.DataFrame({"K": k_range, "Silhouette Score": skor_silhouette})
        st.dataframe(hasil_k, use_container_width=True)
        best_k = int(hasil_k.loc[hasil_k["Silhouette Score"].idxmax(), "K"])
        st.success(f"K terbaik berdasarkan silhouette score tertinggi: **{best_k}**")
        st.session_state["best_k"] = best_k

    st.subheader("3. Jalankan Clustering Final")
    default_k = st.session_state.get("best_k", 5)
    k_final = st.number_input(
        "Pilih jumlah cluster (K) final:", min_value=2, max_value=20, value=default_k, step=1
    )

    if st.button("Jalankan Clustering & Profiling"):
        kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        df_result = df.copy()
        df_result["Cluster"] = labels

        skor_final = silhouette_score(X, labels)
        st.metric("Silhouette Score", f"{skor_final:.4f}")

        st.dataframe(df_result, use_container_width=True)

        st.subheader("4. Profiling per Cluster")
        profil_options = categorical_cols if categorical_cols else all_columns
        profil_col = st.selectbox(
            "Pilih kolom kategorikal untuk profiling tiap cluster:", options=profil_options
        )

        st.write("Jumlah data per cluster:")
        st.dataframe(df_result["Cluster"].value_counts().sort_index())

        for i in sorted(df_result["Cluster"].unique()):
            isi_cluster = df_result[df_result["Cluster"] == i]
            with st.expander(f"Cluster {i} — Total: {len(isi_cluster)} data"):
                st.dataframe(isi_cluster[profil_col].value_counts())

        profile_pct = (
            pd.crosstab(df_result["Cluster"], df_result[profil_col], normalize="index") * 100
        )
        st.write("Persentase distribusi per Cluster:")
        st.dataframe(profile_pct.round(2), use_container_width=True)

        dominant_category = profile_pct.idxmax(axis=1)
        st.write("Kategori paling dominan di tiap Cluster:")
        st.dataframe(dominant_category.rename("Kategori Dominan"))

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        profile_pct.plot(kind="bar", stacked=True, ax=ax2, colormap="tab20")
        ax2.set_title(f"Profiling '{profil_col}' per Cluster")
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Persentase (%)")
        ax2.legend(title=profil_col, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        st.pyplot(fig2)

        csv_buffer = io.StringIO()
        df_result.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Hasil Clustering (CSV)",
            data=csv_buffer.getvalue(),
            file_name="hasil_clustering.csv",
            mime="text/csv",
        )
else:
    st.info("Silakan upload file CSV atau Excel untuk memulai.")

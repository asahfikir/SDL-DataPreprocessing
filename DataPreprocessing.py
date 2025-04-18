import marimo

__generated_with = "0.12.8"
app = marimo.App(layout_file="layouts/DataPreprocessing.grid.json")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Data Preprocessing
        ## By: Rijalul Fikri [2024171004]
        Pada notebook ini kita akan mengolah data `market_sample.csv`. Menerapkan data preprocessing serta Teknik Dimentionality Reduction menggunakan PCA.
        """
    )
    return


@app.cell
def _():
    # Load library yang dibutuhkan
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    return KMeans, PCA, StandardScaler, TSNE, mo, np, pd, plt


@app.cell
def _(mo, pd):
    # Let's load the data
    df = pd.read_csv("market_sample.csv")
    mo.md(
        f"""
        ## Load Data CSV
        Kita akan menggunakan library python yang bernama `panda` untuk memembuka file `market_sample.csv`. Setelah di load kita akan memanggil method describe untuk menampilkan struktur data dari file csv. Dengan menggunakan method `describe` kita bisa melihat sebaran nilai untuk kolom-kolom numerikal.
        {mo.as_html(df.describe(include='all'))}
        """
    )

    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"""
        ## Tampilkan 5 Data Teratas

        {mo.as_html(df.head())}
        """
    )
    return


@app.cell
def _(df, mo):
    nilai_kosong_lama = df.copy().isnull().sum()

    # drop nilai kosong
    df.dropna(inplace=True)

    # Tampilkan text
    mo.md(
      f"""
      ## Nilai-nilai kosong per kolom:
      {mo.as_html(nilai_kosong_lama)}

      Dari tabel diatas terlihat bahwa ada beberapa nilai kosong namun jumlahnya tidak banyak dan jika kita lihat dari kolom sales yang kosong dapat kita asumsikan karena terjadi kesalahan input. Data seperti itu bisa kita drop saja menggunakan

      `df.dropna(inplace=True)`
      """
    )

    # df.dropna(inplace=True)
    return (nilai_kosong_lama,)


@app.cell
def _(df, mo):
    # butuh kita simpan dulu di variabel lain nilai duplikatnya
    # jika tidak nanti akan dioverride karena marimo cumam membolehkan satu reference
    dirty_df = df.duplicated().sum()

    # Hapus duplikat
    df_clean = df.drop_duplicates()

    mo.md(
        f"""
        ## Periksa apakah ada data duplikat
        Kita akan memeriksa apakah ada baris yang duplikat menggunakan perintah `df.duplicated().sum()`

        {dirty_df}

        ## Hapus baris duplikat
        Dari data diatas dapat kita lihat bahwa terdapat `2 baris` duplikat. Karena cuma sedikit maka akan kita hapus saja.

        Jumlah baris setelah dihapus: {df_clean.duplicated().sum()}
        """
    )
    return df_clean, dirty_df


@app.cell
def _(df, df_clean, mo, plt):
    # import seaborn as sns
    # sns.boxplot(data=df[['Quantity', 'UnitPrice', 'Sales', 'Profit']])

    # Fungsi untuk handle outlier menggunakan metode IQR
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower_bound, upper_bound)

    # Terapkan hanya pada kolom-kolom numerikal
    numerical_cols = ["Quantity", "UnitPrice", "Sales", "Profit"]
    df[numerical_cols] = df_clean[numerical_cols].apply(cap_outliers)

    # Visualisasikan menggunakan boxplot
    plt.figure(figsize=(10, 6))
    df[numerical_cols].boxplot()
    plt.title("Boxplot setelah diterapkan IQR")
    plt.xticks(rotation=45)
    # show_graph = plt.show()
    mo.md(
        f"""
        ## Pengecekan Outlier
        Pada Step ini kita akan memeriksa apakah ada anomali terhadap data, dan jika ditemukan kita akan mengatasinya menggunakan metode IQR.
        {mo.as_html(plt.show())}
        """
    )

    return cap_outliers, numerical_cols


@app.cell
def _(df, mo, pd):
    # Ekstrak dulu data penting dari Order Date, rubah ke datetime yang bisa dibaca python
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])

    # Feature Engineering: Ekstrak Hari, Bulan, dan Tahun
    df["Year"] = df["OrderDate"].dt.year
    df["Month"] = df["OrderDate"].dt.month
    df["Day"] = df["OrderDate"].dt.day

    # Kelompokkan berdasar CustomerID dan pola pembelian
    # Karena bisa jadi ada satu custome membeli banyak produk beberapa kali
    # Jadi kita grouping saja
    customer_data = df.groupby("CustomerID").agg({
        "Quantity": ["sum", "mean", "count"],
        "UnitPrice": ["mean", "median"],
        "Sales": ["sum", "mean"],
        "Profit": ["sum", "mean"],
        "Year": ["nunique"],
        "Month": ["nunique"]
    }).reset_index()

    # Flatten columns
    customer_data.columns = ['_'.join(col).strip() for col in customer_data.columns.values]
    customer_data.rename(columns={"CustomerID_": "CustomerID"}, inplace=True)

    # Pilih fitur numerikal untuk keperluan clustering
    features = customer_data.drop("CustomerID", axis=1)

    # mo.as_html(customer_data)
    mo.as_html(features)
    return customer_data, features


@app.cell
def _(StandardScaler, features, mo):
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    mo.md(
        f"""
        ## Standarisasi Nilai Fitur
        Menggunakan method
        ```
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        ```
        """
    )
    return scaled_features, scaler


@app.cell
def _(KMeans, PCA, mo, np, plt, scaled_features):
    # Klusterisasi menggunakan fitur RAW
    # Cari dua fitur paling penting (berdasar nilai varian tertinggi)
    top_two_features = np.argsort(np.var(scaled_features, axis=0))[-2:]
    raw_x, raw_y = scaled_features[:, top_two_features[0]], scaled_features[:, top_two_features[1]]

    kmeans_raw = KMeans(n_clusters=3, random_state=42)
    clusters_raw = kmeans_raw.fit_predict(scaled_features)

    # Plot raw features
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(raw_x, raw_y, c=clusters_raw, cmap="viridis", alpha=0.6)
    plt.title("Clusters on RAW Features\n(Top 2 by Variance)")
    plt.xlabel(f"Feature {top_two_features[0]}")
    plt.ylabel(f"Feature {top_two_features[1]}")

    # Menggunakan Reduksi Dimensi PCA
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    kmeans_pca = KMeans(n_clusters=3, random_state=42)
    clusters_pca = kmeans_pca.fit_predict(pca_features)

    # Plot PCA features
    plt.subplot(122)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters_pca, cmap="viridis", alpha=0.6)
    plt.title("Clusters on PCA-Reduced Features")
    plt.xlabel("PC1 (%.1f%% variance)" % (pca.explained_variance_ratio_[0]*100))
    plt.ylabel("PC2 (%.1f%% variance)" % (pca.explained_variance_ratio_[1]*100))
    plt.tight_layout()
    # plt.show()

    mo.md(
        f"""
        ## Perbandingan visualisasi sebelum dan sesudah Reduksi Dimensi

        Pada bagian kiri adalah visualisasi sebelum dilakukan reduksi dimensi dan dibagian kanan adalah visualisasi stelah reduksi dimensi menggunakan PCA.

        PCA Variance: {pca.explained_variance_ratio_}
    
        {mo.as_html(plt.show())}
        """
    )
    return (
        clusters_pca,
        clusters_raw,
        kmeans_pca,
        kmeans_raw,
        pca,
        pca_features,
        raw_x,
        raw_y,
        top_two_features,
    )


@app.cell
def _(KMeans, TSNE, mo, plt, scaled_features):
    # Clustering dengan t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(scaled_features)

    kmeans_tsne = KMeans(n_clusters=3, random_state=42)
    clusters_tsne = kmeans_tsne.fit_predict(tsne_result)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters_tsne, cmap="viridis", alpha=0.6)
    plt.title("Clusters After t-SNE (2D)")
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.colorbar(label="Cluster")

    mo.md(
        f"""
        ## Clusterisasi menggunakan t-SNE

        Berikut visualisasi menggunakan t-SNE sebagai metode dimentionality reduction nya.
        {mo.as_html(plt.show())}
        """
    )
    return clusters_tsne, kmeans_tsne, tsne, tsne_result


if __name__ == "__main__":
    app.run()

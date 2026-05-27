# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    Holt
)

from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# CONFIG HALAMAN
# ==========================================

st.set_page_config(
    page_title="Forecasting Barang",
    layout="wide"
)

st.title("Forecasting dan Clustering Barang")

# ==========================================
# UPLOAD FILE
# ==========================================

uploaded_file = st.file_uploader(
    "Upload File Excel",
    type=["xlsx"]
)

# ==========================================
# JIKA FILE SUDAH DIUPLOAD
# ==========================================

if uploaded_file is not None:

    # ==========================================
    # MEMBACA FILE EXCEL
    # ==========================================

    df = pd.read_excel(uploaded_file)

    st.subheader("Data Awal")
    st.dataframe(df.head())

    # ==========================================
    # CEK NAMA KOLOM
    # ==========================================

    st.subheader("Nama Kolom")
    st.write(df.columns)

    # ==========================================
    # UBAH FORMAT TANGGAL
    # ==========================================

    df['tgl_input'] = pd.to_datetime(df['tgl_input'])

    # ==========================================
    # MEMBUAT FORMAT BULAN
    # ==========================================

    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

    # ==========================================
    # MEMBUAT PIVOT TABLE
    # ==========================================

    pivot_table = df.pivot_table(
        index='id_produk',
        columns='Bulan',
        values='keluar',
        aggfunc='sum',
        fill_value=0
    )

    # ==========================================
    # URUTAN BULAN
    # ==========================================

    urutan_bulan = [
        'Jan-23','Feb-23','Mar-23','Apr-23','May-23','Jun-23',
        'Jul-23','Aug-23','Sep-23','Oct-23','Nov-23','Dec-23',
        'Jan-24','Feb-24','Mar-24','Apr-24','May-24','Jun-24',
        'Jul-24','Aug-24','Sep-24','Oct-24','Nov-24','Dec-24'
    ]

    pivot_table = pivot_table.reindex(columns=urutan_bulan)

    st.subheader("Pivot Table Barang Keluar")
    st.dataframe(pivot_table)

    # ==========================================
    # DOWNLOAD PIVOT TABLE
    # ==========================================

    csv_data = pivot_table.to_csv().encode('utf-8')

    st.download_button(
        label="Download Pivot Table",
        data=csv_data,
        file_name='data_barang_keluar.csv',
        mime='text/csv'
    )

    # ==========================================
    # CLUSTERING
    # ==========================================

    st.header("Clustering Produk")

    if 'Total' in pivot_table.columns:
        pivot_table = pivot_table.drop(columns=['Total'])

    pivot_table['Total'] = pivot_table.sum(axis=1)

    filtered_data = pivot_table[pivot_table['Total'] > 1]

    filtered_data = filtered_data.drop(columns=['Total'])

    # ==========================================
    # NORMALISASI DATA
    # ==========================================

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(filtered_data)

    # ==========================================
    # ELBOW METHOD
    # ==========================================

    inertia = []

    K = range(1, 10)

    for k in K:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42
        )

        kmeans.fit(scaled_data)

        inertia.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots(figsize=(8,5))

    ax1.plot(K, inertia, marker='o')

    ax1.set_xlabel('Jumlah Cluster (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Metode Elbow')

    ax1.grid(True)

    st.pyplot(fig1)

    # ==========================================
    # PILIH JUMLAH CLUSTER
    # ==========================================

    jumlah_cluster = st.slider(
        "Pilih Jumlah Cluster",
        min_value=2,
        max_value=10,
        value=3
    )

    # ==========================================
    # K-MEANS CLUSTERING
    # ==========================================

    kmeans = KMeans(
        n_clusters=jumlah_cluster,
        random_state=42
    )

    cluster = kmeans.fit_predict(scaled_data)

    filtered_data['Cluster'] = cluster

    st.subheader("Hasil Clustering")
    st.dataframe(filtered_data.head())

    # ==========================================
    # JUMLAH PRODUK PER CLUSTER
    # ==========================================

    st.subheader("Jumlah Produk per Cluster")

    st.write(
        filtered_data['Cluster'].value_counts()
    )

    # ==========================================
    # RATA-RATA TOTAL PER CLUSTER
    # ==========================================

    filtered_data['Total'] = filtered_data.drop(
        columns=['Cluster']
    ).sum(axis=1)

    cluster_summary = filtered_data.groupby(
        'Cluster'
    )['Total'].mean()

    st.subheader("Rata-rata Total per Cluster")

    st.write(cluster_summary)

    # ==========================================
    # PILIH CLUSTER
    # ==========================================

    st.subheader("Lihat Produk Berdasarkan Cluster")

    pilih_cluster = st.selectbox(
        "Pilih Cluster",
        sorted(filtered_data['Cluster'].unique())
    )

    # ==========================================
    # TAMPILKAN PRODUK DALAM CLUSTER
    # ==========================================

    produk_cluster = filtered_data[
        filtered_data['Cluster'] == pilih_cluster
    ].index.tolist()

    st.write(f"Daftar Produk di Cluster {pilih_cluster}")

    df_produk_cluster = pd.DataFrame({
        'Produk': produk_cluster
    })

    st.dataframe(df_produk_cluster)

    st.write(
        f"Jumlah Produk: {len(produk_cluster)}"
    )

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("Forecasting Barang")

    daftar_produk = filtered_data.index.tolist()

    produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    # ==========================================
    # AMBIL DATA PRODUK
    # ==========================================

    data_produk = filtered_data.loc[produk]

    kolom_hapus = []

    if 'Cluster' in data_produk.index:
        kolom_hapus.append('Cluster')

    if 'Total' in data_produk.index:
        kolom_hapus.append('Total')

    data_produk = data_produk.drop(kolom_hapus)

    # ubah ke numerik
    data_produk = pd.to_numeric(data_produk)

    # ==========================================
    # INDEX TANGGAL
    # ==========================================

    data_produk.index = pd.date_range(
        start='2023-01-01',
        periods=len(data_produk),
        freq='ME'
    )

    # ==========================================
    # PILIH METODE FORECASTING
    # ==========================================

    metode = st.selectbox(
        "Pilih Metode Forecasting",
        [
            "Holt-Winters",
            "Double Exponential Smoothing",
            "ARIMA",
            "Perbandingan Semua Metode"
        ]
    )

    # ==========================================
    # JUMLAH FORECAST
    # ==========================================

    jumlah_forecast = st.slider(
        "Jumlah Forecast Bulan",
        min_value=1,
        max_value=12,
        value=6
    )

    # ==========================================
    # VISUALISASI DATA AKTUAL
    # ==========================================

    fig2, ax2 = plt.subplots(figsize=(12,5))

    ax2.plot(
        data_produk.index,
        data_produk.values,
        marker='o',
        label='Data Aktual'
    )

    ax2.set_title(f'Data Aktual Produk {produk}')

    ax2.set_xlabel('Periode')

    ax2.set_ylabel('Jumlah Barang Keluar')

    ax2.legend()

    ax2.grid(True)

    st.pyplot(fig2)

    # ==========================================
    # HOLT-WINTERS
    # ==========================================

    if metode == "Holt-Winters":

        model_hw = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit_hw = model_hw.fit()

        forecast_hw = fit_hw.forecast(
            jumlah_forecast
        )

        forecast_hw = forecast_hw.clip(lower=0)

        st.subheader("Hasil Forecast Holt-Winters")

        st.write(forecast_hw)

        mae_hw = mean_absolute_error(
            data_produk,
            fit_hw.fittedvalues
        )

        st.write(f"MAE Holt-Winters: {mae_hw:.2f}")

        fig3, ax3 = plt.subplots(figsize=(12,5))

        ax3.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax3.plot(
            forecast_hw.index,
            forecast_hw.values,
            marker='o',
            linestyle='--',
            label='Forecast Holt-Winters'
        )

        ax3.set_title(
            f'Forecast Holt-Winters Produk {produk}'
        )

        ax3.legend()

        ax3.grid(True)

        st.pyplot(fig3)

    # ==========================================
    # DOUBLE EXPONENTIAL SMOOTHING
    # ==========================================

    elif metode == "Double Exponential Smoothing":

        model_des = Holt(
            data_produk
        )

        fit_des = model_des.fit()

        forecast_des = fit_des.forecast(
            jumlah_forecast
        )

        forecast_des = forecast_des.clip(lower=0)

        st.subheader(
            "Hasil Forecast Double Exponential Smoothing"
        )

        st.write(forecast_des)

        mae_des = mean_absolute_error(
            data_produk,
            fit_des.fittedvalues
        )

        st.write(
            f"MAE Double Exponential Smoothing: "
            f"{mae_des:.2f}"
        )

        fig4, ax4 = plt.subplots(figsize=(12,5))

        ax4.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax4.plot(
            forecast_des.index,
            forecast_des.values,
            marker='o',
            linestyle='--',
            label='Double Exponential Smoothing'
        )

        ax4.set_title(
            f'Forecast DES Produk {produk}'
        )

        ax4.legend()

        ax4.grid(True)

        st.pyplot(fig4)

    # ==========================================
    # ARIMA
    # ==========================================

    elif metode == "ARIMA":

        model_arima = ARIMA(
            data_produk,
            order=(1,1,1)
        )

        fit_arima = model_arima.fit()

        forecast_arima = fit_arima.forecast(
            steps=jumlah_forecast
        )

        forecast_arima = forecast_arima.clip(lower=0)

        st.subheader("Hasil Forecast ARIMA")

        st.write(forecast_arima)

        fitted_arima = fit_arima.predict(
            start=1,
            end=len(data_produk)-1
        )

        actual_arima = data_produk[1:]

        mae_arima = mean_absolute_error(
            actual_arima,
            fitted_arima
        )

        st.write(f"MAE ARIMA: {mae_arima:.2f}")

        fig5, ax5 = plt.subplots(figsize=(12,5))

        ax5.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax5.plot(
            forecast_arima.index,
            forecast_arima.values,
            marker='o',
            linestyle='--',
            label='Forecast ARIMA'
        )

        ax5.set_title(
            f'Forecast ARIMA Produk {produk}'
        )

        ax5.legend()

        ax5.grid(True)

        st.pyplot(fig5)

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        # ==========================================
        # HOLT-WINTERS
        # ==========================================

        model_hw = ExponentialSmoothing(
            data_produk,
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )

        fit_hw = model_hw.fit()

        forecast_hw = fit_hw.forecast(
            jumlah_forecast
        )

        forecast_hw = forecast_hw.clip(lower=0)

        mae_hw = mean_absolute_error(
            data_produk,
            fit_hw.fittedvalues
        )

        # ==========================================
        # DOUBLE EXPONENTIAL SMOOTHING
        # ==========================================

        model_des = Holt(
            data_produk
        )

        fit_des = model_des.fit()

        forecast_des = fit_des.forecast(
            jumlah_forecast
        )

        forecast_des = forecast_des.clip(lower=0)

        mae_des = mean_absolute_error(
            data_produk,
            fit_des.fittedvalues
        )

        # ==========================================
        # ARIMA
        # ==========================================

        model_arima = ARIMA(
            data_produk,
            order=(1,1,1)
        )

        fit_arima = model_arima.fit()

        forecast_arima = fit_arima.forecast(
            steps=jumlah_forecast
        )

        forecast_arima = forecast_arima.clip(lower=0)

        fitted_arima = fit_arima.predict(
            start=1,
            end=len(data_produk)-1
        )

        actual_arima = data_produk[1:]

        mae_arima = mean_absolute_error(
            actual_arima,
            fitted_arima
        )

        # ==========================================
        # TABEL PERBANDINGAN
        # ==========================================

        st.subheader(
            "Perbandingan Semua Metode Forecasting"
        )

        perbandingan = pd.DataFrame({
            'Metode': [
                'Holt-Winters',
                'Double Exponential Smoothing',
                'ARIMA'
            ],
            'MAE': [
                mae_hw,
                mae_des,
                mae_arima
            ]
        })

        st.dataframe(perbandingan)

        # ==========================================
        # METODE TERBAIK
        # ==========================================

        metode_terbaik = perbandingan.loc[
            perbandingan['MAE'].idxmin()
        ]

        st.success(
            f"Metode terbaik adalah "
            f"{metode_terbaik['Metode']} "
            f"dengan MAE "
            f"{metode_terbaik['MAE']:.2f}"
        )

            # ==========================================
    # HASIL NUMERIK FORECAST
    # ==========================================

    st.subheader("Hasil Forecast Secara Numerik")

    df_hasil_forecast = pd.DataFrame({
        'Periode': forecast_arima.index.strftime('%b-%Y'),
        'Hasil Forecast': forecast_arima.values.round(2)
    })

    st.dataframe(df_hasil_forecast)

    # ==========================================
    # TOTAL FORECAST
    # ==========================================

    total_forecast = forecast_arima.sum()

    rata_forecast = forecast_arima.mean()

    maks_forecast = forecast_arima.max()

    minimum_forecast = forecast_arima.min()

    st.subheader("Statistik Forecast")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Forecast",
            f"{total_forecast:.2f}"
        )

        st.metric(
            "Rata-rata Forecast",
            f"{rata_forecast:.2f}"
        )

    with col2:

        st.metric(
            "Forecast Tertinggi",
            f"{maks_forecast:.2f}"
        )

        st.metric(
            "Forecast Terendah",
            f"{minimum_forecast:.2f}"
        )

    # ==========================================
    # PERBANDINGAN AKTUAL VS FITTED
    # ==========================================

    st.subheader("Perbandingan Aktual vs Prediksi")

    df_perbandingan = pd.DataFrame({
        'Aktual': data_produk_arima.values,
        'Prediksi': fitted_values_arima.values
    })

    df_perbandingan.index = data_produk_arima.index.strftime(
        '%b-%Y'
    )

    df_perbandingan['Selisih'] = (
        df_perbandingan['Aktual']
        - df_perbandingan['Prediksi']
    )

    st.dataframe(
        df_perbandingan.round(2)
    )

    # ==========================================
    # EVALUASI MODEL
    # ==========================================

    st.subheader("Evaluasi Model ARIMA")

    # MAE

    mae = abs(
        fitted_values_arima
        - data_produk_arima
    ).mean()

    # MSE

    mse = (
        (
            fitted_values_arima
            - data_produk_arima
        ) ** 2
    ).mean()

    # RMSE

    rmse = mse ** 0.5

    # MAPE

    mape = (
        abs(
            (
                data_produk_arima
                - fitted_values_arima
            ) / data_produk_arima.replace(0, 1)
        ).mean()
    ) * 100

    # ==========================================
    # TAMPILKAN METRIK
    # ==========================================

    col3, col4 = st.columns(2)

    with col3:

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

    with col4:

        st.metric(
            "MSE",
            f"{mse:.2f}"
        )

        st.metric(
            "MAPE",
            f"{mape:.2f}%"
        )

    # ==========================================
    # KESIMPULAN SEDERHANA
    # ==========================================

    st.subheader("Kesimpulan Forecast")

    st.write(
        f"""
        Berdasarkan hasil forecasting menggunakan metode ARIMA ({p},{d},{q}),
        produk {produk_arima} diperkirakan memiliki total permintaan sebesar
        {total_forecast:.2f} selama {jumlah_forecast_arima} bulan ke depan.
        """
    )

        # ==========================================
        # VISUALISASI PERBANDINGAN
        # ==========================================

fig6, ax6 = plt.subplots(figsize=(12,5))

        ax6.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Data Aktual'
        )

        ax6.plot(
            forecast_hw.index,
            forecast_hw.values,
            marker='o',
            linestyle='--',
            label='Holt-Winters'
        )

        ax6.plot(
            forecast_des.index,
            forecast_des.values,
            marker='o',
            linestyle='--',
            label='Double Exponential Smoothing'
        )

        ax6.plot(
            forecast_arima.index,
            forecast_arima.values,
            marker='o',
            linestyle='--',
            label='ARIMA'
        )

        ax6.set_title(
            f'Perbandingan Forecast Produk {produk}'
        )

        ax6.set_xlabel('Periode')

        ax6.set_ylabel('Jumlah Barang Keluar')

        ax6.legend()

        ax6.grid(True)

        st.pyplot(fig6)

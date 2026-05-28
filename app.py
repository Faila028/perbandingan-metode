# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    Holt
)

from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# FUNGSI CROSTON
# ==========================================

def croston(ts, alpha=0.4, n_forecast=6):

    ts = np.array(ts)

    demand = []
    interval = []

    last_demand = 0
    last_interval = 1

    forecast = []

    q = 1

    for i in range(len(ts)):

        if ts[i] > 0:

            last_demand = alpha * ts[i] + (1 - alpha) * last_demand

            last_interval = alpha * q + (1 - alpha) * last_interval

            q = 1

        else:

            q += 1

        if last_interval != 0:
            forecast.append(last_demand / last_interval)
        else:
            forecast.append(0)

    future_forecast = [forecast[-1]] * n_forecast

    return np.array(forecast), np.array(future_forecast)

# ==========================================
# CONFIG HALAMAN
# ==========================================

st.set_page_config(
    page_title="Forecasting Barang",
    page_icon="📦",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
    text-align: center;
    font-weight: bold;
}

h2, h3 {
    color: #1f4e79;
}

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stDownloadButton>button {
    background-color: #28a745;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

[data-testid="stMetricValue"] {
    color: #1f77b4;
    font-size: 28px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# JUDUL
# ==========================================

st.title("📦 Clustering dan Forecasting Permintaan Barang")

st.markdown("""
### Sistem Analisis Barang Keluar
Aplikasi ini digunakan untuk:
- Clustering produk menggunakan K-Means
- Forecasting permintaan barang
- Perbandingan metode forecasting
""")

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

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Awal")

    st.dataframe(df.head())

    # ==========================================
    # DASHBOARD
    # ==========================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Data",
            len(df)
        )

    with col2:
        st.metric(
            "Jumlah Produk",
            df['id_produk'].nunique()
        )

    with col3:
        st.metric(
            "Total Barang Keluar",
            int(df['keluar'].sum())
        )

    # ==========================================
    # FORMAT TANGGAL
    # ==========================================

    df['tgl_input'] = pd.to_datetime(df['tgl_input'])

    df['Bulan'] = df['tgl_input'].dt.strftime('%b-%y')

    # ==========================================
    # PIVOT TABLE
    # ==========================================

    pivot_table = df.pivot_table(
        index='id_produk',
        columns='Bulan',
        values='keluar',
        aggfunc='sum',
        fill_value=0
    )

    urutan_bulan = [
        'Jan-23','Feb-23','Mar-23','Apr-23','May-23','Jun-23',
        'Jul-23','Aug-23','Sep-23','Oct-23','Nov-23','Dec-23',
        'Jan-24','Feb-24','Mar-24','Apr-24','May-24','Jun-24',
        'Jul-24','Aug-24','Sep-24','Oct-24','Nov-24','Dec-24'
    ]

    pivot_table = pivot_table.reindex(
        columns=urutan_bulan
    )

    st.subheader("📊 Pivot Table Barang Keluar")

    st.dataframe(pivot_table)

    # ==========================================
    # DOWNLOAD
    # ==========================================

    csv_data = pivot_table.to_csv().encode('utf-8')

    st.download_button(
        label="⬇️ Download Pivot Table",
        data=csv_data,
        file_name='data_barang_keluar.csv',
        mime='text/csv'
    )

    # ==========================================
    # CLUSTERING
    # ==========================================

    st.header("📌 Clustering Produk")

    pivot_table['Total'] = pivot_table.sum(axis=1)

    filtered_data = pivot_table[
        pivot_table['Total'] > 1
    ]

    filtered_data = filtered_data.drop(
        columns=['Total']
    )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        filtered_data
    )

    # ==========================================
    # ELBOW
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

    ax1.plot(
        K,
        inertia,
        marker='o'
    )

    ax1.set_title("Metode Elbow")

    ax1.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    st.pyplot(fig1)

    # ==========================================
    # PILIH CLUSTER
    # ==========================================

    jumlah_cluster = st.slider(
        "Pilih Jumlah Cluster",
        2,
        10,
        3
    )

    kmeans = KMeans(
        n_clusters=jumlah_cluster,
        random_state=42
    )

    cluster = kmeans.fit_predict(
        scaled_data
    )

    filtered_data['Cluster'] = cluster

    st.subheader("📌 Hasil Clustering")

    st.dataframe(filtered_data.head())

    # ==========================================
    # FORECASTING
    # ==========================================

    st.header("📈 Forecasting Barang")

    daftar_produk = filtered_data.index.tolist()

    produk = st.selectbox(
        "Pilih Produk",
        daftar_produk
    )

    data_produk = filtered_data.loc[produk]

    kolom_hapus = []

    if 'Cluster' in data_produk.index:
        kolom_hapus.append('Cluster')

    data_produk = data_produk.drop(
        kolom_hapus
    )

    data_produk = pd.to_numeric(
        data_produk
    )

    data_produk.index = pd.date_range(
        start='2023-01-01',
        periods=len(data_produk),
        freq='ME'
    )

    metode = st.selectbox(
        "Pilih Metode Forecasting",
        [
            "Holt-Winters",
            "Double Exponential Smoothing",
            "ARIMA",
            "Croston",
            "Perbandingan Semua Metode"
        ]
    )

    jumlah_forecast = st.slider(
        "Jumlah Forecast Bulan",
        1,
        12,
        6
    )

    # ==========================================
    # DATA AKTUAL
    # ==========================================

    fig2, ax2 = plt.subplots(
        figsize=(12,5)
    )

    ax2.plot(
        data_produk.index,
        data_produk.values,
        marker='o',
        label='Data Aktual'
    )

    ax2.set_title(
        f'Data Aktual Produk {produk}'
    )

    ax2.legend()

    ax2.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

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

        mae_hw = mean_absolute_error(
            data_produk,
            fit_hw.fittedvalues
        )

        st.metric(
            "MAE Holt-Winters",
            f"{mae_hw:.2f}"
        )

        fig3, ax3 = plt.subplots(figsize=(12,5))

        ax3.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Aktual'
        )

        ax3.plot(
            forecast_hw.index,
            forecast_hw.values,
            marker='o',
            linestyle='--',
            label='Forecast HW'
        )

        ax3.legend()

        ax3.grid(True)

        st.pyplot(fig3)

    # ==========================================
    # DES
    # ==========================================

    elif metode == "Double Exponential Smoothing":

        model_des = Holt(
            data_produk
        )

        fit_des = model_des.fit()

        forecast_des = fit_des.forecast(
            jumlah_forecast
        )

        mae_des = mean_absolute_error(
            data_produk,
            fit_des.fittedvalues
        )

        st.metric(
            "MAE DES",
            f"{mae_des:.2f}"
        )

        fig4, ax4 = plt.subplots(figsize=(12,5))

        ax4.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Aktual'
        )

        ax4.plot(
            forecast_des.index,
            forecast_des.values,
            marker='o',
            linestyle='--',
            label='Forecast DES'
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

        fitted_arima = fit_arima.predict(
            start=1,
            end=len(data_produk)-1
        )

        actual_arima = data_produk[1:]

        mae_arima = mean_absolute_error(
            actual_arima,
            fitted_arima
        )

        st.metric(
            "MAE ARIMA",
            f"{mae_arima:.2f}"
        )

        fig5, ax5 = plt.subplots(figsize=(12,5))

        ax5.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Aktual'
        )

        ax5.plot(
            forecast_arima.index,
            forecast_arima.values,
            marker='o',
            linestyle='--',
            label='Forecast ARIMA'
        )

        ax5.legend()

        ax5.grid(True)

        st.pyplot(fig5)

    # ==========================================
    # CROSTON
    # ==========================================

    elif metode == "Croston":

        fitted_croston, forecast_croston = croston(
            data_produk.values,
            alpha=0.4,
            n_forecast=jumlah_forecast
        )

        future_index = pd.date_range(
            start=data_produk.index[-1] + pd.offsets.MonthEnd(1),
            periods=jumlah_forecast,
            freq='ME'
        )

        mae_croston = mean_absolute_error(
            data_produk.values,
            fitted_croston
        )

        st.metric(
            "MAE Croston",
            f"{mae_croston:.2f}"
        )

        fig6, ax6 = plt.subplots(figsize=(12,5))

        ax6.plot(
            data_produk.index,
            data_produk.values,
            marker='o',
            label='Aktual'
        )

        ax6.plot(
            future_index,
            forecast_croston,
            marker='o',
            linestyle='--',
            label='Forecast Croston'
        )

        ax6.legend()

        ax6.grid(True)

        st.pyplot(fig6)

    # ==========================================
    # PERBANDINGAN SEMUA METODE
    # ==========================================

    else:

        # HW

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

        mae_hw = mean_absolute_error(
            data_produk,
            fit_hw.fittedvalues
        )

        # DES

        model_des = Holt(
            data_produk
        )

        fit_des = model_des.fit()

        forecast_des = fit_des.forecast(
            jumlah_forecast
        )

        mae_des = mean_absolute_error(
            data_produk,
            fit_des.fittedvalues
        )

        # ARIMA

        model_arima = ARIMA(
            data_produk,
            order=(1,1,1)
        )

        fit_arima = model_arima.fit()

        forecast_arima = fit_arima.forecast(
            steps=jumlah_forecast
        )

        fitted_arima = fit_arima.predict(
            start=1,
            end=len(data_produk)-1
        )

        actual_arima = data_produk[1:]

        mae_arima = mean_absolute_error(
            actual_arima,
            fitted_arima
        )

        # CROSTON

        fitted_croston, forecast_croston = croston(
            data_produk.values,
            alpha=0.4,
            n_forecast=jumlah_forecast
        )

        mae_croston = mean_absolute_error(
            data_produk.values,
            fitted_croston
        )

        # ==========================================
        # TABEL
        # ==========================================

        st.subheader(
            "📊 Perbandingan Metode"
        )

        perbandingan = pd.DataFrame({
            'Metode': [
                'Holt-Winters',
                'DES',
                'ARIMA',
                'Croston'
            ],
            'MAE': [
                mae_hw,
                mae_des,
                mae_arima,
                mae_croston
            ]
        })

        st.dataframe(
            perbandingan.style.highlight_min(
                axis=0,
                color='lightgreen'
            )
        )

        # ==========================================
        # GRAFIK MAE
        # ==========================================

        fig_mae, ax_mae = plt.subplots(
            figsize=(8,5)
        )

        ax_mae.bar(
            perbandingan['Metode'],
            perbandingan['MAE']
        )

        ax_mae.set_title(
            'Perbandingan Nilai MAE'
        )

        ax_mae.grid(True)

        st.pyplot(fig_mae)

        # ==========================================
        # METODE TERBAIK
        # ==========================================

        metode_terbaik = perbandingan.loc[
            perbandingan['MAE'].idxmin()
        ]

        st.balloons()

        st.success(
            f"Metode terbaik adalah "
            f"{metode_terbaik['Metode']} "
            f"dengan MAE "
            f"{metode_terbaik['MAE']:.2f}"
        )

import streamlit as st
import pandas as pd
import joblib
import requests
import folium
import plotly.express as px

from streamlit_folium import st_folium

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Prediksi Risiko Banjir Indonesia",
    page_icon="🌊",
    layout="wide"
)

# =====================================================
# LOAD FILE
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("model_rf_banjir_indonesia.pkl")

@st.cache_data
def load_master():
    return pd.read_csv("master_kecamatan.csv")

model = load_model()
master = load_master()

# =====================================================
# HEADER
# =====================================================

st.title("🌊 Sistem Prediksi Risiko Banjir Indonesia")
st.markdown("""
Prediksi risiko banjir tingkat kecamatan menggunakan algoritma
**Random Forest** berdasarkan variabel meteorologi dan geospasial.
""")

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📍 Pilih Lokasi")

kabupaten = st.sidebar.selectbox(
    "Kabupaten/Kota",
    sorted(master["NAME_2"].unique())
)

data_kab = master[
    master["NAME_2"] == kabupaten
]

kecamatan = st.sidebar.selectbox(
    "Kecamatan",
    sorted(data_kab["NAME_3"].unique())
)

lokasi = data_kab[
    data_kab["NAME_3"] == kecamatan
].iloc[0]

# =====================================================
# DATA WILAYAH
# =====================================================

lat = float(lokasi["lat"])
lon = float(lokasi["long"])

elevation = float(lokasi["elevation"])
ndvi = float(lokasi["ndvi"])
slope = float(lokasi["slope"])
landcover = int(lokasi["landcover_class"])

# =====================================================
# LAYOUT
# =====================================================

col1, col2 = st.columns([1, 2])

# =====================================================
# INFO WILAYAH
# =====================================================

with col1:

    st.subheader("📊 Karakteristik Wilayah")

    st.metric("Elevasi", f"{elevation:.2f} m")
    st.metric("NDVI", f"{ndvi:.3f}")
    st.metric("Slope", f"{slope:.3f}")
    st.metric("Landcover Class", landcover)

    st.write(f"Latitude : {lat}")
    st.write(f"Longitude : {lon}")

# =====================================================
# MAP
# =====================================================

with col2:

    st.subheader("🗺️ Lokasi Kecamatan")

    m = folium.Map(
        location=[lat, lon],
        zoom_start=11
    )

    folium.Marker(
        [lat, lon],
        popup=f"{kecamatan}",
        tooltip=f"{kecamatan}"
    ).add_to(m)

    st_folium(
        m,
        width=900,
        height=450
    )

st.divider()

# =====================================================
# FORECAST BUTTON
# =====================================================

if st.button(
    "📡 Ambil Forecast dan Prediksi 7 Hari",
    use_container_width=True
):

    with st.spinner("Mengambil data cuaca dari Open-Meteo..."):

        try:

            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}"
                f"&longitude={lon}"
                f"&daily=temperature_2m_mean,precipitation_sum"
                f"&forecast_days=7"
                f"&timezone=Asia%2FJakarta"
            )

            response = requests.get(url)

            data = response.json()

            hasil_prediksi = []

            for i in range(7):

                tanggal = data["daily"]["time"][i]

                avg_temperature = float(
                    data["daily"]["temperature_2m_mean"][i]
                )

                avg_rainfall = float(
                    data["daily"]["precipitation_sum"][i]
                )

                # pendekatan sederhana
                max_rainfall = avg_rainfall

                # menggunakan rata-rata dataset
                soil_moisture = 36.30

                X_pred = pd.DataFrame({

                    "avg_rainfall":[avg_rainfall],

                    "max_rainfall":[max_rainfall],

                    "avg_temperature":[avg_temperature],

                    "elevation":[elevation],

                    "landcover_class":[landcover],

                    "ndvi":[ndvi],

                    "slope":[slope],

                    "soil_moisture":[soil_moisture]

                })

                pred = model.predict(X_pred)[0]

                prob = (
                    model.predict_proba(X_pred)[0][1]
                    * 100
                )

                status = (
                    "Risiko Tinggi"
                    if pred == 1
                    else "Risiko Rendah"
                )

                hasil_prediksi.append({

                    "Tanggal": tanggal,

                    "Curah Hujan (mm)":
                        round(avg_rainfall, 2),

                    "Temperatur (°C)":
                        round(avg_temperature, 2),

                    "Probabilitas (%)":
                        round(prob, 2),

                    "Status":
                        status

                })

            hasil_df = pd.DataFrame(
                hasil_prediksi
            )

            st.success(
                f"Forecast berhasil diproses untuk Kecamatan {kecamatan}"
            )

            # =====================================================
            # TABEL
            # =====================================================

            st.subheader(
                "📅 Prediksi Risiko Banjir 7 Hari Ke Depan"
            )

            st.dataframe(
                hasil_df,
                use_container_width=True
            )

            # =====================================================
            # GRAFIK
            # =====================================================

            fig = px.line(
                hasil_df,
                x="Tanggal",
                y="Probabilitas (%)",
                markers=True,
                title="Tren Probabilitas Risiko Banjir"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =====================================================
            # STATUS HARI TERTINGGI
            # =====================================================

            risiko_tertinggi = hasil_df.loc[
                hasil_df["Probabilitas (%)"].idxmax()
            ]

            st.subheader(
                "🚨 Risiko Tertinggi"
            )

            if risiko_tertinggi["Probabilitas (%)"] >= 50:

                st.error(
                    f"""
                    Tanggal: {risiko_tertinggi['Tanggal']}

                    Probabilitas: {risiko_tertinggi['Probabilitas (%)']}%

                    Status: {risiko_tertinggi['Status']}
                    """
                )

            else:

                st.success(
                    f"""
                    Risiko banjir relatif rendah.

                    Probabilitas tertinggi:
                    {risiko_tertinggi['Probabilitas (%)']}%
                    """
                )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan: {e}"
            )
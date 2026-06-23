import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Simulator Kebijakan Keuntungan Toko",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CACHE MODEL
# =====================================================
@st.cache_resource
def load_model():

    # Data historis sederhana
    X_train = np.array([
        [5, 10],
        [10, 20],
        [15, 5],
        [20, 25],
        [25, 15]
    ])

    y_train = np.array([
        50,
        80,
        110,
        90,
        150
    ])

    model = LinearRegression().fit(X_train, y_train)

    return model

@st.cache_data
def buat_kurva_respons(diskon):

    data = []

    for nilai_iklan in range(0, 55, 5):

        prediksi = model.predict(
            np.array([[nilai_iklan, diskon]])
        )[0]

        data.append(
            [nilai_iklan, prediksi]
        )

    return pd.DataFrame(
        data,
        columns=[
            "Anggaran Iklan",
            "Keuntungan"
        ]
    )

# =====================================================
# LOAD MODEL
# =====================================================
model = load_model()

# =====================================================
# BASELINE
# =====================================================
baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]
BASELINE = {
    "iklan": 10,
    "diskon": 10
}

# =====================================================
# FUNGSI SIMULASI
# =====================================================
def run_simulation(new_iklan, new_diskon):

    intervention_input = np.array([
        [new_iklan, new_diskon]
    ])

    prediction = model.predict(
        intervention_input
    )[0]

    delta_y = prediction - baseline_pred

    return prediction, delta_y

def kembali_ke_baseline():
    st.session_state["iklan"] = BASELINE["iklan"]
    st.session_state["diskon"] = BASELINE["diskon"]

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.header-box{
    background: linear-gradient(90deg,#0f172a,#1e3a8a);
    padding:25px;
    border-radius:15px;
    color:white;
    text-align:center;
}

.footer-box{
    margin-top:40px;
    padding:15px;
    text-align:center;
    border-radius:10px;
    background:#0f172a;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header-box">
<h1>🚀 Simulator Kebijakan Keuntungan Toko</h1>
<p>Analisis What-If Berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)
st.header("Analisis Skenario What-If")
st.write(
    "Gunakan slider pada sidebar untuk menguji berbagai skenario kebijakan secara real-time."
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🎛️ Panel Intervensi")

st.sidebar.markdown(
    """
    Ubah nilai variabel berikut untuk
    mensimulasikan dampak kebijakan.
    """
)

iklan_slider = st.sidebar.slider(
    "Anggaran Iklan (Juta)",
    0,
    50,
    BASELINE["iklan"],
    key="iklan"
)

diskon_slider = st.sidebar.slider(
    "Besaran Diskon (%)",
    0,
    50,
    BASELINE["diskon"],
    key="diskon"
)

st.sidebar.button(
    "🔄 Kembalikan ke Baseline",
    on_click=kembali_ke_baseline,
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Setiap perubahan slider akan langsung
    menghitung ulang prediksi keuntungan.
    """
)

# =====================================================
# VALIDASI BASELINE
# =====================================================
st.subheader("📌 Kondisi Saat Ini (Baseline)")

st.info(
    f"""
    Baseline digunakan sebagai pembanding.

    • Anggaran Iklan : Rp 10 Juta

    • Diskon : 10%

    • Prediksi Keuntungan :
      Rp {baseline_pred:.2f} Juta
    """
)

# =====================================================
# SIMULASI
# =====================================================
hasil_pred, delta = run_simulation(
    iklan_slider,
    diskon_slider
)

# =====================================================
# HASIL SIMULASI
# =====================================================
st.subheader("📊 Hasil Simulasi")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Prediksi Keuntungan",
        f"Rp {hasil_pred:.2f} Jt",
        f"{delta:.2f} Jt"
    )

with col2:

    st.metric(
        "Baseline",
        f"Rp {baseline_pred:.2f} Jt"
    )

with col3:

    perubahan = (
        delta / baseline_pred
    ) * 100

    st.metric(
        "Persentase Perubahan",
        f"{perubahan:.2f}%"
    )

with col4:

    if delta > 0:
        status = "📈 Positif"
    elif delta < 0:
        status = "📉 Negatif"
    else:
        status = "➖ Netral"

    st.metric(
        "Status Skenario",
        status
    )

with st.expander(
    "📈 Lihat Kurva Respons Anggaran Iklan"
):

    kurva = buat_kurva_respons(
        diskon_slider
    )

    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.plot(
        kurva["Anggaran Iklan"],
        kurva["Keuntungan"],
        marker="o"
    )

    ax2.set_title(
        "Sensitivity Analysis"
    )

    ax2.set_xlabel(
        "Anggaran Iklan (Juta)"
    )

    ax2.set_ylabel(
        "Prediksi Keuntungan (Juta)"
    )

    ax2.grid(True)

    st.pyplot(fig2)

    plt.close(fig2)

    st.caption(
        "Diskon dipertahankan sesuai skenario aktif."
    )
    
# =====================================================
# STORYTELLING
# =====================================================
st.subheader("📝 Narasi Hasil")

if delta > 20:

    st.success(
        f"""
        Strategi ini meningkatkan keuntungan
        sebesar Rp {delta:.2f} Juta.

        Dampaknya sangat positif dan layak
        dipertimbangkan sebagai kebijakan bisnis.
        """
    )

elif delta > 0:

    st.info(
        f"""
        Strategi memberikan peningkatan
        keuntungan sebesar Rp {delta:.2f} Juta.

        Dampaknya positif namun masih moderat.
        """
    )

elif delta == 0:

    st.warning(
        """
        Tidak ada perubahan dibandingkan
        kondisi baseline.
        """
    )

else:

    st.error(
        f"""
        Strategi menyebabkan penurunan
        keuntungan sebesar Rp {abs(delta):.2f} Juta.

        Kebijakan ini kurang direkomendasikan.
        """
    )

# =====================================================
# VISUALISASI
# =====================================================
st.subheader("📈 Grafik Perbandingan")

data_plot = pd.DataFrame({
    "Skenario": [
        "Baseline",
        "Intervensi"
    ],
    "Keuntungan": [
        baseline_pred,
        hasil_pred
    ]
})

fig, ax = plt.subplots(figsize=(6,4))

ax.bar(
    data_plot["Skenario"],
    data_plot["Keuntungan"]
)

ax.set_title(
    "Perbandingan Baseline vs Intervensi"
)

ax.set_ylabel(
    "Keuntungan (Juta)"
)

st.pyplot(fig)
plt.close(fig)

st.subheader("📉 Delta Analysis Chart")

delta_df = pd.DataFrame({
    "Kategori": ["Perubahan"],
    "Delta": [delta]
})

fig_delta, ax_delta = plt.subplots()

ax_delta.bar(
    delta_df["Kategori"],
    delta_df["Delta"]
)

ax_delta.set_ylabel("Delta Keuntungan (Juta)")
ax_delta.set_title("Delta dari Baseline")

st.pyplot(fig_delta)

plt.close(fig_delta)

st.subheader("📋 Detail Kebijakan")

detail = pd.DataFrame({

    "Variabel": [
        "Anggaran Iklan",
        "Diskon"
    ],

    "Baseline": [
        "10 Juta",
        "10%"
    ],

    "Skenario": [
        f"{iklan_slider} Juta",
        f"{diskon_slider}%"
    ]

})

st.dataframe(
    detail,
    use_container_width=True
)

# =====================================================
# DATAFRAME
# =====================================================
st.subheader("📋 Ringkasan Data")

st.dataframe(
    data_plot,
    use_container_width=True
)

# =====================================================
# DELTA ANALYSIS
# =====================================================
st.subheader("🎯 Delta Analysis")

if abs(delta) > 30:

    st.warning(
        """
        Sistem sangat sensitif terhadap
        perubahan variabel kebijakan.
        """
    )

elif abs(delta) > 10:

    st.info(
        """
        Sistem cukup sensitif terhadap
        perubahan variabel kebijakan.
        """
    )

else:

    st.success(
        """
        Sistem relatif stabil terhadap
        perubahan variabel kebijakan.
        """
    )

st.subheader("📑 Rekomendasi Kebijakan")

if delta > 0:

    st.success(
        f"""
        Berdasarkan simulasi, peningkatan anggaran iklan
        menjadi {iklan_slider} juta dan diskon {diskon_slider}%
        diperkirakan meningkatkan keuntungan sebesar
        Rp {delta:.2f} juta dibanding kondisi saat ini.

        Kebijakan ini layak dipertimbangkan.
        """
    )

else:

    st.warning(
        f"""
        Skenario yang dipilih belum memberikan
        keuntungan yang lebih baik dibanding baseline.

        Disarankan mencoba kombinasi kebijakan lain.
        """
    )
    
# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<hr>

<div style='text-align:center;'>

<h3>
Universitas Nusantara PGRI Kediri
</h3>

<p>
Pemodelan dan Simulasi Berbasis Machine Learning
</p>

<p>
Simulator Interaktif dan Analisis What-If
</p>

<p>
Developed by Azmi Sophia Wakova 🚀
</p>

</div>
""",
unsafe_allow_html=True)
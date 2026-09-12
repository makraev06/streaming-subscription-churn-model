import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import FeatureEngineeringTransformer, get_risk_tier

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Streaming Churn Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. DESAIN TAMPILAN BERSIH & MINIMALIS (LIGHT MODE)
# ==============================================================================
bg_main = "#F8FAFC"
bg_card = "#FFFFFF"
border_color = "#E2E8F0"
text_main = "#0F172A"
text_muted = "#64748B"
accent_primary = "#4F46E5"
accent_primary_hover = "#4338CA"
tab_bg = "#F1F5F9"
tab_active_bg = "#FFFFFF"
plotly_theme = "plotly_white"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {{
        background-color: {bg_main};
        color: {text_main};
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }}

    /* Card Box Minimalis */
    .clean-card {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
    }}
    .clean-card:hover {{
        border-color: {accent_primary};
    }}

    /* Header & Typography */
    .app-header {{
        margin-bottom: 22px;
        padding-bottom: 14px;
        border-bottom: 1px solid {border_color};
    }}
    .app-title {{
        font-size: 1.75rem;
        font-weight: 700;
        color: {text_main};
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }}
    .app-subtitle {{
        font-size: 0.95rem;
        color: {text_muted};
        margin: 0;
        line-height: 1.5;
    }}

    /* Stat & KPI Cards */
    .stat-box {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }}
    .stat-label {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {text_muted};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }}
    .stat-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {text_main};
        margin: 0;
    }}
    .stat-desc {{
        font-size: 0.78rem;
        color: {text_muted};
        margin-top: 2px;
    }}

    /* Badge Risk Simple */
    .risk-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }}
    .risk-high {{
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #F87171;
    }}
    .risk-med {{
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FBBF24;
    }}
    .risk-low {{
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #4ADE80;
    }}

    /* Action Item */
    .action-item {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-left: 3px solid {accent_primary};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }}
    .action-title {{
        font-size: 0.85rem;
        font-weight: 700;
        color: {accent_primary};
        margin-bottom: 4px;
    }}
    .action-desc {{
        font-size: 0.88rem;
        color: {text_main};
        margin: 0;
        line-height: 1.45;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {tab_bg};
        padding: 4px;
        border-radius: 10px;
        border: 1px solid {border_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 38px;
        background-color: transparent;
        border-radius: 8px;
        color: {text_muted};
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {tab_active_bg} !important;
        color: {text_main} !important;
        font-weight: 600 !important;
        border: 1px solid {border_color} !important;
    }}

    /* Button Primary */
    .stButton > button {{
        background-color: {accent_primary};
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 8px 18px;
        transition: background-color 0.15s ease;
    }}
    .stButton > button:hover {{
        background-color: {accent_primary_hover};
        color: #FFFFFF;
    }}

    /* Custom Form & Labels */
    label {{
        color: {text_main} !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. RESOURCE LOADERS & CACHING
# ==============================================================================
@st.cache_resource
def load_model_pipeline():
    model_path = os.path.join('models', 'churn_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_model_metrics():
    metrics_path = os.path.join('models', 'metrics.pkl')
    if os.path.exists(metrics_path):
        return joblib.load(metrics_path)
    return {'roc_auc': 0.9339, 'f1': 0.8519, 'recall': 0.8565, 'accuracy': 0.8471, 'precision': 0.8475}

pipeline = load_model_pipeline()
metrics = load_model_metrics()

# Ambil nama algoritma yang digunakan
model_algo_name = "Gradient Boosting Classifier"
if pipeline is not None and hasattr(pipeline, 'steps'):
    model_algo_name = type(pipeline.steps[-1][1]).__name__.replace("Classifier", "")

# ==============================================================================
# 4. SIDEBAR - INFORMASI SISTEM & STATUS MODEL
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
        <div style="background: #EEF2FF; width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
            🎵
        </div>
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #0F172A;">Churn Predictor</h3>
            <p style="margin: 0; font-size: 0.75rem; color: #64748B;">Retention Intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Widget Status Model Khusus
    status_color = "#059669" if pipeline is not None else "#DC2626"
    status_bg = "#DCFCE7" if pipeline is not None else "#FEE2E2"
    status_text = "● ONLINE" if pipeline is not None else "● OFFLINE"
    
    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; margin: 12px 0 16px 0;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 4px;">
            <span style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight:700;">Status Model</span>
            <span style="font-size: 0.75rem; color: {status_color}; font-weight:700; background: {status_bg}; padding: 2px 8px; border-radius: 12px;">{status_text}</span>
        </div>
        <div style="font-size: 0.92rem; font-weight: 700; color: #0F172A;">{model_algo_name}</div>
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 2px;">Artefak: <code>churn_model.pkl</code></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Metrik Validasi")
    st.markdown(f"""
    - **ROC-AUC Score:** `{metrics['roc_auc']:.4f}`
    - **F1-Score:** `{metrics['f1']:.4f}`
    - **Recall (Churn):** `{metrics['recall']:.4f}`
    - **Akurasi:** `{metrics['accuracy']:.4f}`
    """)

    st.markdown("---")
    st.markdown("### 👤 Informasi Proyek")
    st.markdown("""
    - **Peserta:** Soni
    - **Program:** GDG Machine Learning
    - **Tahun:** 2026
    """)

# ==============================================================================
# 5. HEADER UTAMA (SIMPEL & BERSIH)
# ==============================================================================
st.markdown("""
<div class="app-header">
    <div style="display: inline-block; background: #DCFCE7; color: #15803D; font-size: 0.78rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; border: 1px solid #86EFAC; margin-bottom: 8px;">
        ● Model Siap Digunakan v1.0
    </div>
    <h1 class="app-title">Streaming Subscription Churn Predictor</h1>
    <p class="app-subtitle">
        Analisis probabilitas pelanggan musik berhenti berlangganan berdasarkan profil penggunaan, interaksi, dan riwayat akun.
    </p>
</div>
""", unsafe_allow_html=True)

# Navigasi Tab
tab_single, tab_batch, tab_insights = st.tabs([
    "Prediksi Tunggal",
    "Analisis Batch (CSV)",
    "Kinerja & Wawasan Model"
])

# ==============================================================================
# 6. TAB 1: PREDIKSI TUNGGAL (SINGLE CUSTOMER)
# ==============================================================================
with tab_single:
    # Baris Preset Contoh Profil
    preset_cols = st.columns([1, 1, 1])
    with preset_cols[0]:
        if st.button("🔴 Muat Contoh Profil Risiko Tinggi", use_container_width=True):
            st.session_state['age'] = 58
            st.session_state['location'] = 'New York'
            st.session_state['sub_type'] = 'Free'
            st.session_state['payment_plan'] = 'Monthly'
            st.session_state['payment_method'] = 'Debit Card'
            st.session_state['tenure'] = 60
            st.session_state['weekly_hours'] = 4.0
            st.session_state['avg_session'] = 12.0
            st.session_state['skip_rate'] = 0.85
            st.session_state['weekly_songs'] = 25
            st.session_state['unique_songs'] = 10
            st.session_state['fav_artists'] = 2
            st.session_state['friends'] = 5
            st.session_state['playlists'] = 1
            st.session_state['shared'] = 0
            st.session_state['pauses'] = 3
            st.session_state['notif_clicks'] = 2
            st.session_state['cs_inquiries'] = 'High'
            st.rerun()

    with preset_cols[1]:
        if st.button("🟢 Muat Contoh Profil Pelanggan Setia", use_container_width=True):
            st.session_state['age'] = 28
            st.session_state['location'] = 'California'
            st.session_state['sub_type'] = 'Premium'
            st.session_state['payment_plan'] = 'Yearly'
            st.session_state['payment_method'] = 'Credit Card'
            st.session_state['tenure'] = 730
            st.session_state['weekly_hours'] = 35.0
            st.session_state['avg_session'] = 75.0
            st.session_state['skip_rate'] = 0.15
            st.session_state['weekly_songs'] = 380
            st.session_state['unique_songs'] = 220
            st.session_state['fav_artists'] = 35
            st.session_state['friends'] = 120
            st.session_state['playlists'] = 45
            st.session_state['shared'] = 28
            st.session_state['pauses'] = 0
            st.session_state['notif_clicks'] = 32
            st.session_state['cs_inquiries'] = 'Low'
            st.rerun()

    with preset_cols[2]:
        if st.button("🔄 Reset ke Nilai Awal", use_container_width=True):
            for k in ['age', 'location', 'sub_type', 'payment_plan', 'payment_method', 'tenure', 'weekly_hours', 'avg_session', 'skip_rate', 'weekly_songs', 'unique_songs', 'fav_artists', 'friends', 'playlists', 'shared', 'pauses', 'notif_clicks', 'cs_inquiries']:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Profil & Langganan")
            age = st.slider("Usia", 18, 80, st.session_state.get('age', 34))
            
            location_options = ['California', 'New York', 'Washington', 'Florida', 'Texas', 'Montana',
                                'New Jersey', 'Georgia', 'Wisconsin', 'Idaho', 'Alabama', 'South Carolina',
                                'North Carolina', 'Utah', 'West Virginia', 'Maine', 'Nebraska', 'Virginia',
                                'Vermont', 'North Dakota']
            default_loc_idx = location_options.index(st.session_state.get('location', 'California')) if st.session_state.get('location', 'California') in location_options else 0
            location = st.selectbox("Lokasi", location_options, index=default_loc_idx)
            
            sub_options = ['Free', 'Premium', 'Family', 'Student']
            default_sub_idx = sub_options.index(st.session_state.get('sub_type', 'Premium')) if st.session_state.get('sub_type', 'Premium') in sub_options else 1
            sub_type = st.selectbox("Tipe Langganan", sub_options, index=default_sub_idx)
            
            plan_options = ['Monthly', 'Yearly']
            default_plan_idx = plan_options.index(st.session_state.get('payment_plan', 'Monthly')) if st.session_state.get('payment_plan', 'Monthly') in plan_options else 0
            payment_plan = st.selectbox("Paket Pembayaran", plan_options, index=default_plan_idx)
            
            method_options = ['Credit Card', 'Paypal', 'Debit Card', 'Apple Pay']
            default_method_idx = method_options.index(st.session_state.get('payment_method', 'Credit Card')) if st.session_state.get('payment_method', 'Credit Card') in method_options else 0
            payment_method = st.selectbox("Metode Pembayaran", method_options, index=default_method_idx)
            
            tenure_input = st.number_input("Lama Bergabung (Hari)", 1, 3000, st.session_state.get('tenure', 365))

        with col2:
            st.markdown("#### Aktivitas Mendengarkan")
            weekly_hours = st.slider("Jam Dengar Mingguan", 0.0, 50.0, float(st.session_state.get('weekly_hours', 22.0)), 0.5)
            avg_session = st.slider("Rata-rata Durasi Sesi (Menit)", 1.0, 120.0, float(st.session_state.get('avg_session', 50.0)), 1.0)
            skip_rate = st.slider("Rasio Skip Lagu", 0.0, 1.0, float(st.session_state.get('skip_rate', 0.35)), 0.01)
            weekly_songs = st.number_input("Total Lagu Mingguan", 1, 500, int(st.session_state.get('weekly_songs', 240)))
            unique_songs = st.number_input("Total Lagu Unik Mingguan", 1, 300, int(st.session_state.get('unique_songs', 145)))

        with col3:
            st.markdown("#### Keterlibatan & Dukungan")
            fav_artists = st.slider("Jumlah Artis Favorit", 0, 50, int(st.session_state.get('fav_artists', 18)))
            friends = st.slider("Jumlah Teman di Aplikasi", 0, 200, int(st.session_state.get('friends', 65)))
            playlists = st.slider("Playlist Dibuat", 0, 100, int(st.session_state.get('playlists', 20)))
            shared = st.slider("Playlist Dibagikan", 0, 50, int(st.session_state.get('shared', 8)))
            pauses = st.slider("Frekuensi Jeda Akun", 0, 5, int(st.session_state.get('pauses', 1)))
            notif_clicks = st.slider("Klik Notifikasi", 0, 50, int(st.session_state.get('notif_clicks', 18)))
            
            cs_options = ['Low', 'Medium', 'High']
            default_cs_idx = cs_options.index(st.session_state.get('cs_inquiries', 'Medium')) if st.session_state.get('cs_inquiries', 'Medium') in cs_options else 1
            cs_inquiries = st.selectbox("Frekuensi CS Inquiry", cs_options, index=default_cs_idx)

        submitted = st.form_submit_button("Hitung Prediksi Churn", use_container_width=True)

    if submitted:
        if pipeline is None:
            st.error("Model tidak ditemukan di folder `models/churn_model.pkl`.")
        else:
            input_dict = {
                'customer_id': 999999,
                'age': age,
                'location': location,
                'subscription_type': sub_type,
                'payment_plan': payment_plan,
                'num_subscription_pauses': pauses,
                'payment_method': payment_method,
                'customer_service_inquiries': cs_inquiries,
                'signup_date': -int(tenure_input),
                'weekly_hours': float(weekly_hours),
                'average_session_length': float(avg_session),
                'song_skip_rate': float(skip_rate),
                'weekly_songs_played': int(weekly_songs),
                'weekly_unique_songs': int(unique_songs),
                'num_favorite_artists': int(fav_artists),
                'num_platform_friends': int(friends),
                'num_playlists_created': int(playlists),
                'num_shared_playlists': int(shared),
                'notifications_clicked': int(notif_clicks)
            }
            input_df = pd.DataFrame([input_dict])

            with st.spinner("Memproses prediksi..."):
                prob = pipeline.predict_proba(input_df)[:, 1][0]
                risk_tier, risk_icon, risk_label_id = get_risk_tier(prob)

            st.markdown("---")
            st.markdown("### Hasil Prediksi")

            r_col1, r_col2, r_col3 = st.columns([1.2, 1.2, 1.6])

            with r_col1:
                st.markdown(f"""
                <div class="stat-box" style="height: 160px; display: flex; flex-direction: column; justify-content: center;">
                    <div class="stat-label">Probabilitas Churn</div>
                    <div class="stat-value">{prob * 100:.1f}%</div>
                    <div class="stat-desc">Estimasi Risiko Model</div>
                </div>
                """, unsafe_allow_html=True)

            with r_col2:
                risk_class = "risk-high" if risk_tier == "High" else ("risk-med" if risk_tier == "Medium" else "risk-low")
                st.markdown(f"""
                <div class="stat-box" style="height: 160px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <div class="stat-label">Kategori Risiko</div>
                    <div style="margin: 8px 0;">
                        <span class="risk-pill {risk_class}">{risk_icon} Risiko {risk_label_id} ({risk_tier})</span>
                    </div>
                    <div class="stat-desc">Tingkat Prioritas Retensi</div>
                </div>
                """, unsafe_allow_html=True)

            with r_col3:
                gauge_color = "#EF4444" if prob >= 0.7 else ("#F59E0B" if prob >= 0.3 else "#10B981")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': "%", 'font': {'size': 24, 'color': text_main}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': text_muted},
                        'bar': {'color': gauge_color, 'thickness': 0.28},
                        'bgcolor': "rgba(0,0,0,0)",
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': text_main, 'width': 3},
                            'thickness': 0.8,
                            'value': prob * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=160,
                    margin=dict(l=20, r=20, t=25, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': 'Plus Jakarta Sans', 'color': text_main}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Rekomendasi Aksi Bisnis
            st.markdown("#### Rekomendasi Tindakan")
            actions = []
            if risk_tier == "High":
                actions.append(("Prioritas Retensi", "Berikan penawaran diskon khusus perpanjangan atau promo langganan tahunan dengan potongan harga."))
                if cs_inquiries == "High":
                    actions.append(("Follow-up Dukungan CS", "Eskalasi kendala ke tim penanganan pelanggan untuk menyelesaikan masalah layanan."))
                if skip_rate > 0.5:
                    actions.append(("Pembaruan Rekomendasi Musik", "Rasio skip lagu tinggi; tawarkan playlist rekomendasi baru sesuai preferensi artis favorit."))
                if pauses >= 2:
                    actions.append(("Opsi Pause Fleksibel", "Berikan fasilitas jeda akun sementara gratis agar tidak berhenti berlangganan permanen."))
            elif risk_tier == "Medium":
                actions.append(("Peningkatan Keterlibatan", "Kirim notifikasi update album atau playlist mingguan yang relevan."))
                if friends < 10:
                    actions.append(("Fitur Sosial", "Ajak pelanggan menghubungkan teman atau membagikan playlist."))
                if payment_plan == "Monthly":
                    actions.append(("Penawaran Paket Tahunan", "Tawarkan keuntungan upgrade ke paket tahunan."))
            else:
                actions.append(("Pertahankan Loyalitas", "Pelanggan berada dalam kategori sehat. Pertahankan kualitas layanan dan kirim rekomendasi konten berkala."))

            for title, desc in actions:
                st.markdown(f"""
                <div class="action-item">
                    <div class="action-title">• {title}</div>
                    <p class="action-desc">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# 7. TAB 2: ANALISIS BATCH (CSV)
# ==============================================================================
with tab_batch:
    st.markdown("### Analisis File CSV Pelanggan")
    st.markdown("Unggah berkas CSV untuk memprediksi risiko churn banyak pelanggan sekaligus.")

    u_col1, u_col2, u_col3 = st.columns([2.5, 1.2, 1.2])
    with u_col1:
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], label_visibility="collapsed")
    with u_col2:
        load_demo = st.button("Muat 200 Baris Demo", use_container_width=True)
    with u_col3:
        if os.path.exists('data/sample_test_200_customers.csv'):
            with open('data/sample_test_200_customers.csv', 'rb') as f_sample:
                sample_data = f_sample.read()
            st.download_button(
                label="Unduh File Sampel",
                data=sample_data,
                file_name="sample_test_200_customers.csv",
                mime="text/csv",
                use_container_width=True
            )

    df_batch = None
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
    elif load_demo and os.path.exists('data/sample_test_200_customers.csv'):
        df_batch = pd.read_csv('data/sample_test_200_customers.csv')
        st.info("Berhasil memuat 200 baris data dari berkas sampel.")

    if df_batch is not None:
        st.markdown(f"**Pratinjau Data ({len(df_batch):,} Baris):**")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button("Jalankan Prediksi Batch", use_container_width=True):
            if pipeline is None:
                st.error("Model pipeline belum dimuat!")
            else:
                with st.spinner("Menghitung prediksi..."):
                    probs = pipeline.predict_proba(df_batch)[:, 1]
                    res_df = df_batch.copy()
                    res_df['churn_probability'] = np.round(probs, 4)
                    res_df['risk_tier'] = [get_risk_tier(p)[0] for p in probs]
                    res_df['risk_label'] = [get_risk_tier(p)[2] for p in probs]

                st.session_state['batch_results'] = res_df

    if 'batch_results' in st.session_state:
        res_df = st.session_state['batch_results']
        st.success("Analisis selesai.")

        total_cnt = len(res_df)
        high_cnt = (res_df['risk_tier'] == 'High').sum()
        med_cnt = (res_df['risk_tier'] == 'Medium').sum()
        low_cnt = (res_df['risk_tier'] == 'Low').sum()
        avg_prob = res_df['churn_probability'].mean() * 100

        kpi_cols = st.columns(5)
        kpi_cols[0].markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Total Data</div>
            <div class="stat-value">{total_cnt:,}</div>
        </div>
        """, unsafe_allow_html=True)

        kpi_cols[1].markdown(f"""
        <div class="stat-box" style="border-top: 3px solid #EF4444;">
            <div class="stat-label">Risiko Tinggi</div>
            <div class="stat-value" style="color: #EF4444;">{high_cnt:,}</div>
            <div class="stat-desc">{high_cnt/total_cnt*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        kpi_cols[2].markdown(f"""
        <div class="stat-box" style="border-top: 3px solid #F59E0B;">
            <div class="stat-label">Risiko Sedang</div>
            <div class="stat-value" style="color: #F59E0B;">{med_cnt:,}</div>
            <div class="stat-desc">{med_cnt/total_cnt*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        kpi_cols[3].markdown(f"""
        <div class="stat-box" style="border-top: 3px solid #10B981;">
            <div class="stat-label">Risiko Rendah</div>
            <div class="stat-value" style="color: #10B981;">{low_cnt:,}</div>
            <div class="stat-desc">{low_cnt/total_cnt*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        kpi_cols[4].markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Rata-rata Churn</div>
            <div class="stat-value">{avg_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig_pie = px.pie(
                res_df, names='risk_tier',
                title="Proporsi Tingkat Risiko Churn",
                color='risk_tier',
                color_discrete_map={'High': '#EF4444', 'Medium': '#F59E0B', 'Low': '#10B981'},
                hole=0.45,
                template=plotly_theme
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans', 'color': text_main})
            st.plotly_chart(fig_pie, use_container_width=True)

        with v_col2:
            if 'subscription_type' in res_df.columns:
                fig_bar = px.histogram(
                    res_df, x='subscription_type', color='risk_tier',
                    title="Distribusi Risiko Berdasarkan Tipe Langganan",
                    barmode='group',
                    color_discrete_map={'High': '#EF4444', 'Medium': '#F59E0B', 'Low': '#10B981'},
                    template=plotly_theme
                )
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans', 'color': text_main})
                st.plotly_chart(fig_bar, use_container_width=True)

        # Filter dan Tabel
        st.markdown("#### Tabel Hasil Prediksi")
        filter_risk = st.selectbox("Filter berdasarkan kategori risiko:", ["Semua", "Hanya High Risk", "Hanya Medium Risk", "Hanya Low Risk"])
        
        filtered_df = res_df.copy()
        if "High" in filter_risk:
            filtered_df = filtered_df[filtered_df['risk_tier'] == 'High']
        elif "Medium" in filter_risk:
            filtered_df = filtered_df[filtered_df['risk_tier'] == 'Medium']
        elif "Low" in filter_risk:
            filtered_df = filtered_df[filtered_df['risk_tier'] == 'Low']
            
        st.dataframe(filtered_df, use_container_width=True)

        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh Hasil Prediksi (CSV)",
            data=csv_bytes,
            file_name="churn_predictions_export.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==============================================================================
# 8. TAB 3: KINERJA & WAWASAN MODEL
# ==============================================================================
with tab_insights:
    st.markdown("### Evaluasi Model & Penjelasan Fitur")
    st.markdown("Ringkasan performa model yang dievaluasi pada data pengujian independen.")

    m_cols = st.columns(4)
    m_cols[0].markdown(f"""
    <div class="stat-box" style="border-top: 3px solid {accent_primary};">
        <div class="stat-label">ROC-AUC Score</div>
        <div class="stat-value" style="color: {accent_primary};">{metrics['roc_auc']:.4f}</div>
        <div class="stat-desc">Perankingan Probabilitas</div>
    </div>
    """, unsafe_allow_html=True)

    m_cols[1].markdown(f"""
    <div class="stat-box" style="border-top: 3px solid #10B981;">
        <div class="stat-label">F1-Score</div>
        <div class="stat-value" style="color: #10B981;">{metrics['f1']:.4f}</div>
        <div class="stat-desc">Harmoni Precision & Recall</div>
    </div>
    """, unsafe_allow_html=True)

    m_cols[2].markdown(f"""
    <div class="stat-box" style="border-top: 3px solid #EF4444;">
        <div class="stat-label">Recall (Churn)</div>
        <div class="stat-value" style="color: #EF4444;">{metrics['recall']:.4f}</div>
        <div class="stat-desc">Deteksi Kasus Churn</div>
    </div>
    """, unsafe_allow_html=True)

    m_cols[3].markdown(f"""
    <div class="stat-box" style="border-top: 3px solid #F59E0B;">
        <div class="stat-label">Akurasi</div>
        <div class="stat-value" style="color: #F59E0B;">{metrics['accuracy']:.4f}</div>
        <div class="stat-desc">Akurasi Keseluruhan</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 15 Fitur Paling Berpengaruh (Feature Importance)")

    if pipeline is not None and hasattr(pipeline, 'steps'):
        try:
            clf = pipeline.steps[-1][1]
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                preproc = pipeline.named_steps.get('preprocessor', None)
                feature_names = []
                if preproc is not None:
                    for name, trans, cols in preproc.transformers_:
                        last = trans.steps[-1][1] if hasattr(trans, 'steps') else trans
                        if hasattr(last, 'get_feature_names_out'):
                            feature_names.extend(last.get_feature_names_out(cols))
                        else:
                            feature_names.extend(cols)

                if len(feature_names) != len(importances):
                    feature_names = [f"Fitur {i+1}" for i in range(len(importances))]

                fi_df = pd.DataFrame({
                    'Fitur': feature_names[:len(importances)],
                    'Importance': importances
                }).sort_values('Importance', ascending=True).tail(15)

                fig_fi = px.bar(
                    fi_df, x='Importance', y='Fitur', orientation='h',
                    template=plotly_theme,
                    color='Importance',
                    color_continuous_scale=['#818CF8', '#4F46E5']
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'family': 'Plus Jakarta Sans', 'color': text_main},
                    height=480,
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.info("Informasi kontribusi fitur sedang dimuat...")

    st.markdown("#### Pertimbangan Bisnis: False Positive vs False Negative")
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown(f"""
        <div class="clean-card" style="border-left: 3px solid #EF4444;">
            <div style="font-weight: 700; color: #EF4444; margin-bottom: 6px;">Dampak False Negative (FN)</div>
            <p style="font-size: 0.88rem; color: {text_main}; margin: 0; line-height: 1.45;">
                Pelanggan diprediksi <b>aman</b> tetapi sebenarnya <b>berhenti berlangganan</b>. 
                Hal ini menyebabkan kehilangan pendapatan tanpa kesempatan melakukan upaya retensi. Model diprioritaskan menjaga <b>Recall {metrics['recall']*100:.1f}%</b> untuk menekan risiko ini.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b_col2:
        st.markdown(f"""
        <div class="clean-card" style="border-left: 3px solid #F59E0B;">
            <div style="font-weight: 700; color: #F59E0B; margin-bottom: 6px;">Dampak False Positive (FP)</div>
            <p style="font-size: 0.88rem; color: {text_main}; margin: 0; line-height: 1.45;">
                Pelanggan diprediksi <b>akan churn</b> padahal sebenarnya <b>tetap setia</b>. 
                Dampaknya adalah pengeluaran promo/insentif yang tidak perlu. Keseimbangan ini dijaga dengan metrik <b>Precision {metrics['precision']*100:.1f}%</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

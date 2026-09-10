import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from src.preprocessing import FeatureEngineeringTransformer, get_risk_tier

# ==============================================================================
# 1. PAGE CONFIGURATION & SYSTEM META
# ==============================================================================
st.set_page_config(
    page_title="StreamPulse AI | Churn Intelligence & Retention Studio",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. VIBRANT & BRIGHT MODERN SAAS DESIGN SYSTEM (CSS)
# ==============================================================================
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Color Tokens (Bright & Vibrant Light Palette) */
    :root {
        --bg-base: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-card: #E2E8F0;
        --accent-indigo: #4F46E5;
        --accent-violet: #7C3AED;
        --accent-sky: #0284C7;
        --accent-emerald: #059669;
        --accent-amber: #D97706;
        --accent-rose: #E11D48;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-primary);
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Vibrant Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 45%, #F0FDF4 100%);
        border: 1px solid #C7D2FE;
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.08), 0 8px 10px -6px rgba(79, 70, 229, 0.04);
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #1E1B4B 0%, #4338CA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #475569;
        max-width: 800px;
        line-height: 1.6;
        margin: 0;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #DCFCE7;
        border: 1px solid #86EFAC;
        color: #15803D;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }

    /* Clean Bright Cards */
    .bright-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
        margin-bottom: 18px;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .bright-card:hover {
        border-color: #C7D2FE;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.08);
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* KPI Summary Stat Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Risk Badges (Vibrant Colors) */
    .badge-risk {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.01em;
    }

    .badge-risk-high {
        background: #FEE2E2;
        border: 1px solid #FCA5A5;
        color: #B91C1C;
    }

    .badge-risk-medium {
        background: #FEF3C7;
        border: 1px solid #FCD34D;
        color: #B45309;
    }

    .badge-risk-low {
        background: #DCFCE7;
        border: 1px solid #86EFAC;
        color: #15803D;
    }

    /* Retention Action Playbook Box */
    .playbook-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid var(--accent-indigo);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        display: flex;
        gap: 14px;
        align-items: flex-start;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    }

    .playbook-tag {
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
    }

    .playbook-text {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 10px;
        color: #64748B;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 0 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        border: 1px solid #E2E8F0 !important;
    }

    /* Primary Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
        transform: translateY(-1px);
    }
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

# ==============================================================================
# 4. SIDEBAR BRANDING & SPECS
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <div style="background: linear-gradient(135deg, #4F46E5, #7C3AED); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; color: white; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);">
            🎧
        </div>
        <div>
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #0F172A;">StreamPulse AI</h3>
            <p style="margin: 0; font-size: 0.78rem; color: #64748B;">Retention Intelligence Engine</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Engine Status Widget
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px; margin-bottom: 20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 4px;">
            <span style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight:700;">Status Model</span>
            <span style="font-size: 0.75rem; color: #059669; font-weight:700; background: #DCFCE7; padding: 2px 8px; border-radius: 12px;">● ONLINE</span>
        </div>
        <div style="font-size: 0.9rem; font-weight: 700; color: #0F172A;">Random Forest (Pipeline)</div>
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 2px;">Artefak: <code>churn_model.pkl</code></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Metrik Validasi Model")
    st.markdown(f"""
    - **ROC-AUC Score:** `{metrics['roc_auc']:.4f}`
    - **F1-Score:** `{metrics['f1']:.4f}`
    - **Recall (Churn):** `{metrics['recall']:.4f}`
    - **Akurasi Global:** `{metrics['accuracy']:.4f}`
    """)
    
    st.markdown("---")
    st.markdown("### 👤 Pengembang")
    st.markdown("""
    - **Nama Peserta:** Soni
    - **Program:** GDG Final Project Machine Learning
    - **Tahun Proyek:** 2026
    """)

# ==============================================================================
# 5. HERO HEADER BANNER
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🟢 Model Siap Produksi v1.0 Aktif</div>
    <h1 class="hero-title">Streaming Subscription Churn Predictor</h1>
    <p class="hero-subtitle">
        Platform cerdas Machine Learning untuk memprediksi probabilitas berhenti berlangganan, mengidentifikasi anomali perilaku mendengar, dan mengeksekusi strategi retensi pelanggan secara presisi.
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_single, tab_batch, tab_insights = st.tabs([
    "🔮 Single Customer Studio",
    "📁 Batch CSV Intelligence",
    "📊 Model Analytics & Explainability"
])

# ==============================================================================
# 6. TAB 1: SINGLE CUSTOMER RETENTION STUDIO
# ==============================================================================
with tab_single:
    st.markdown("### 👤 Prediksi & Rencana Retensi Pelanggan Tunggal")
    st.markdown("Sesuaikan parameter profil dan aktivitas di bawah ini untuk melihat estimasi probabilitas churn secara *real-time*.")

    # Preset Quick Load Buttons
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("🔴 Muat Contoh Profil Risiko Tinggi (High Churn)", use_container_width=True):
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

    with preset_col2:
        if st.button("🟢 Muat Contoh Profil Pelanggan Setia (Low Risk)", use_container_width=True):
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

    with preset_col3:
        if st.button("🔄 Reset ke Nilai Default", use_container_width=True):
            for k in ['age', 'location', 'sub_type', 'payment_plan', 'payment_method', 'tenure', 'weekly_hours', 'avg_session', 'skip_rate', 'weekly_songs', 'unique_songs', 'fav_artists', 'friends', 'playlists', 'shared', 'pauses', 'notif_clicks', 'cs_inquiries']:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    with st.form("single_prediction_studio"):
        col_prof, col_stream, col_social = st.columns(3)

        with col_prof:
            st.markdown("""
            <div class="card-title" style="color: #4F46E5;">
                <span>👤</span> Profil & Paket Langganan
            </div>
            """, unsafe_allow_html=True)
            age = st.slider("Usia Pelanggan", 18, 80, st.session_state.get('age', 34))
            location_options = ['California', 'New York', 'Washington', 'Florida', 'Texas', 'Montana',
                                'New Jersey', 'Georgia', 'Wisconsin', 'Idaho', 'Alabama', 'South Carolina',
                                'North Carolina', 'Utah', 'West Virginia', 'Maine', 'Nebraska', 'Virginia',
                                'Vermont', 'North Dakota']
            default_loc_idx = location_options.index(st.session_state.get('location', 'California')) if st.session_state.get('location', 'California') in location_options else 0
            location = st.selectbox("Wilayah Geografis", location_options, index=default_loc_idx)
            
            sub_options = ['Free', 'Premium', 'Family', 'Student']
            default_sub_idx = sub_options.index(st.session_state.get('sub_type', 'Premium')) if st.session_state.get('sub_type', 'Premium') in sub_options else 1
            sub_type = st.selectbox("Tipe Langganan", sub_options, index=default_sub_idx)
            
            plan_options = ['Monthly', 'Yearly']
            default_plan_idx = plan_options.index(st.session_state.get('payment_plan', 'Monthly')) if st.session_state.get('payment_plan', 'Monthly') in plan_options else 0
            payment_plan = st.selectbox("Paket Pembayaran", plan_options, index=default_plan_idx)
            
            method_options = ['Credit Card', 'Paypal', 'Debit Card', 'Apple Pay']
            default_method_idx = method_options.index(st.session_state.get('payment_method', 'Credit Card')) if st.session_state.get('payment_method', 'Credit Card') in method_options else 0
            payment_method = st.selectbox("Metode Pembayaran", method_options, index=default_method_idx)
            
            tenure_input = st.number_input("Masa Aktif Akun (Hari Sejak Bergabung)", 1, 3000, st.session_state.get('tenure', 365), help="Jumlah hari sejak pelanggan mendaftar akun")

        with col_stream:
            st.markdown("""
            <div class="card-title" style="color: #7C3AED;">
                <span>🎧</span> Perilaku Mendengarkan Musik
            </div>
            """, unsafe_allow_html=True)
            weekly_hours = st.slider("Jam Dengar Mingguan (Jam)", 0.0, 50.0, float(st.session_state.get('weekly_hours', 22.0)), 0.5)
            avg_session = st.slider("Rata-rata Durasi Sesi (Menit)", 1.0, 120.0, float(st.session_state.get('avg_session', 50.0)), 1.0)
            skip_rate = st.slider("Rasio Melewati Lagu (Song Skip Rate)", 0.0, 1.0, float(st.session_state.get('skip_rate', 0.35)), 0.01, help="Proporsi lagu yang dilewati sebelum selesai diputar")
            weekly_songs = st.number_input("Total Lagu Diputar Mingguan", 1, 500, int(st.session_state.get('weekly_songs', 240)))
            unique_songs = st.number_input("Total Lagu Unik Mingguan", 1, 300, int(st.session_state.get('unique_songs', 145)))

        with col_social:
            st.markdown("""
            <div class="card-title" style="color: #059669;">
                <span>💬</span> Keterlibatan Sosial & Dukungan
            </div>
            """, unsafe_allow_html=True)
            fav_artists = st.slider("Jumlah Artis Favorit", 0, 50, int(st.session_state.get('fav_artists', 18)))
            friends = st.slider("Jumlah Teman di Platform", 0, 200, int(st.session_state.get('friends', 65)))
            playlists = st.slider("Playlist yang Dibuat", 0, 100, int(st.session_state.get('playlists', 20)))
            shared = st.slider("Playlist yang Dibagikan", 0, 50, int(st.session_state.get('shared', 8)))
            pauses = st.slider("Frekuensi Jeda Langganan (Subscription Pauses)", 0, 5, int(st.session_state.get('pauses', 1)))
            notif_clicks = st.slider("Notifikasi Promosi yang Diklik", 0, 50, int(st.session_state.get('notif_clicks', 18)))
            
            cs_options = ['Low', 'Medium', 'High']
            default_cs_idx = cs_options.index(st.session_state.get('cs_inquiries', 'Medium')) if st.session_state.get('cs_inquiries', 'Medium') in cs_options else 1
            cs_inquiries = st.selectbox("Frekuensi Inquiry Customer Service", cs_options, index=default_cs_idx)

        submitted = st.form_submit_button("⚡ Analisis & Prediksi Risiko Pelanggan", use_container_width=True)

    if submitted:
        if pipeline is None:
            st.error("⚠️ Model belum dimuat. Pastikan file `models/churn_model.pkl` tersedia.")
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

            with st.spinner("Mengevaluasi pipeline & inferensi probabilitas..."):
                prob = pipeline.predict_proba(input_df)[:, 1][0]
                risk_tier, risk_icon, risk_label_id = get_risk_tier(prob)

            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("### 📊 Hasil Prediksi & Penilaian Risiko")

            res_col1, res_col2, res_col3 = st.columns([1.2, 1.2, 1.6])

            with res_col1:
                st.markdown("""
                <div class="bright-card" style="text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center;">
                    <span class="kpi-label">Probabilitas Churn</span>
                    <h1 style="font-size: 2.8rem; font-weight: 800; margin: 4px 0; color: #0F172A;">{:.1f}%</h1>
                    <span style="font-size: 0.8rem; color: #64748B;">Confidence Score Model ML</span>
                </div>
                """.format(prob * 100), unsafe_allow_html=True)

            with res_col2:
                badge_class = "badge-risk-high" if risk_tier == "High" else ("badge-risk-medium" if risk_tier == "Medium" else "badge-risk-low")
                st.markdown(f"""
                <div class="bright-card" style="text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <span class="kpi-label">Klasifikasi Risiko</span>
                    <div style="margin: 10px 0;">
                        <span class="badge-risk {badge_class}">{risk_icon} {risk_tier} Risk ({risk_label_id})</span>
                    </div>
                    <span style="font-size: 0.8rem; color: #64748B;">Prioritas Penanganan Retensi</span>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': "%", 'font': {'size': 26, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                        'bar': {'color': "#E11D48" if prob >= 0.7 else ("#D97706" if prob >= 0.3 else "#059669"), 'thickness': 0.28},
                        'bgcolor': "#F1F5F9",
                        'steps': [
                            {'range': [0, 30], 'color': "#DCFCE7"},
                            {'range': [30, 70], 'color': "#FEF3C7"},
                            {'range': [70, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "#0F172A", 'width': 3},
                            'thickness': 0.8,
                            'value': prob * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=180,
                    margin=dict(l=20, r=20, t=25, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': 'Plus Jakarta Sans'}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Tailored Business Action Plan
            st.markdown("### 🎯 Rekomendasi Strategi Retensi Khusus (Actionable Playbook)")

            playbooks = []
            if risk_tier == "High":
                playbooks.append(("PRIORITAS RETENSI TINGGI", "Tawarkan promo diskon perpanjangan loyalitas 30% atau upgrade ke paket Tahunan dengan bonus 2 bulan gratis."))
                if cs_inquiries == "High":
                    playbooks.append(("CS ESCALATION", "Tugaskan Senior Customer Care Specialist untuk menghubungi pelanggan secara proaktif dan menyelesaikan keluhan akun."))
                if skip_rate > 0.5:
                    playbooks.append(("CONTENT RE-ALIGNMENT", "Rasio skip tinggi mengindikasikan ketidaksesuaian konten. Rekomendasikan kurasi playlist harian baru berbasis artis favorit."))
                if pauses >= 2:
                    playbooks.append(("PAUSE PREVENTION", "Berikan fleksibilitas freeze akun sementara tanpa biaya selama 30 hari untuk mencegah pembatalan permanen."))
            elif risk_tier == "Medium":
                playbooks.append(("PERSONALIZED ENGAGEMENT", "Kirimkan push notification terpersonalisasi mengenai konser online atau rilis album eksklusif artis favorit."))
                if friends < 10:
                    playbooks.append(("SOCIAL ONBOARDING", "Ajak pelanggan menggunakan fitur 'Friend Activity' dan bagikan playlist ke media sosial untuk meningkatkan keterikatan."))
                if payment_plan == "Monthly":
                    playbooks.append(("PLAN UPSELL", "Tawarkan promosi hemat beralih ke paket Tahunan dengan keuntungan reward eksklusif."))
            else:
                playbooks.append(("LOYALTY PRESERVATION", "Pelanggan berada dalam kondisi sangat sehat. Pertahankan kualitas streaming tinggi dan berikan undangan uji coba fitur beta."))

            for tag, text in playbooks:
                st.markdown(f"""
                <div class="playbook-card">
                    <span class="playbook-tag">{tag}</span>
                    <p class="playbook-text">{text}</p>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# 7. TAB 2: BATCH CSV INTELLIGENCE
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Prediction Intelligence (CSV File)")
    st.markdown("Proses data pelanggan dalam skala besar secara otomatis untuk audit churn massal dan perencanaan kampanye retensi bulanan.")

    upload_col1, upload_col2, upload_col3 = st.columns([2.5, 1.2, 1.2])
    with upload_col1:
        uploaded_file = st.file_uploader("Unggah Berkas CSV Pelanggan", type=["csv"])
    with upload_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_demo = st.button("⚡ Muat Langsung 200 Baris Demo", use_container_width=True)
    with upload_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if os.path.exists('data/sample_test_200_customers.csv'):
            with open('data/sample_test_200_customers.csv', 'rb') as f_sample:
                sample_data = f_sample.read()
            st.download_button(
                label="📥 Unduh File Sampel (200 CSV)",
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
        st.info("Memuat 200 baris sampel dari `data/sample_test_200_customers.csv`.")


    if df_batch is not None:
        st.markdown(f"**Pratinjau Data ({len(df_batch):,} Baris):**")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button("🚀 Jalankan Analisis Prediksi Batch", use_container_width=True):
            if pipeline is None:
                st.error("Model pipeline belum dimuat!")
            else:
                with st.spinner("Menghitung probabilitas churn untuk seluruh baris..."):
                    probs = pipeline.predict_proba(df_batch)[:, 1]
                    res_df = df_batch.copy()
                    res_df['churn_probability'] = np.round(probs, 4)
                    res_df['risk_tier'] = [get_risk_tier(p)[0] for p in probs]
                    res_df['risk_label'] = [get_risk_tier(p)[2] for p in probs]

                st.session_state['batch_results'] = res_df

    if 'batch_results' in st.session_state:
        res_df = st.session_state['batch_results']
        st.success("✅ Pemrosesan Batch Selesai dengan Sukses!")

        # Batch KPI Summary
        total_cnt = len(res_df)
        high_cnt = (res_df['risk_tier'] == 'High').sum()
        med_cnt = (res_df['risk_tier'] == 'Medium').sum()
        low_cnt = (res_df['risk_tier'] == 'Low').sum()
        avg_prob = res_df['churn_probability'].mean() * 100

        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Total Pelanggan</div>
                <div class="kpi-value">{total_cnt:,}</div>
            </div>
            <div class="kpi-card" style="border-left: 4px solid #E11D48;">
                <div class="kpi-label">High Risk</div>
                <div class="kpi-value" style="color: #E11D48;">{high_cnt:,}</div>
                <div class="kpi-delta" style="color: #E11D48;">{high_cnt/total_cnt*100:.1f}% Total</div>
            </div>
            <div class="kpi-card" style="border-left: 4px solid #D97706;">
                <div class="kpi-label">Medium Risk</div>
                <div class="kpi-value" style="color: #D97706;">{med_cnt:,}</div>
                <div class="kpi-delta" style="color: #D97706;">{med_cnt/total_cnt*100:.1f}% Total</div>
            </div>
            <div class="kpi-card" style="border-left: 4px solid #059669;">
                <div class="kpi-label">Low Risk</div>
                <div class="kpi-value" style="color: #059669;">{low_cnt:,}</div>
                <div class="kpi-delta" style="color: #059669;">{low_cnt/total_cnt*100:.1f}% Total</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Rata-rata Churn</div>
                <div class="kpi-value">{avg_prob:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Batch Visualizations
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig_pie = px.pie(
                res_df, names='risk_tier',
                title="Proporsi Tingkat Risiko Churn",
                color='risk_tier',
                color_discrete_map={'High': '#E11D48', 'Medium': '#F59E0B', 'Low': '#10B981'},
                hole=0.45
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans'})
            st.plotly_chart(fig_pie, use_container_width=True)

        with v_col2:
            if 'subscription_type' in res_df.columns:
                fig_bar = px.histogram(
                    res_df, x='subscription_type', color='risk_tier',
                    title="Distribusi Risiko Berdasarkan Tipe Langganan",
                    barmode='group',
                    color_discrete_map={'High': '#E11D48', 'Medium': '#F59E0B', 'Low': '#10B981'}
                )
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans'})
                st.plotly_chart(fig_bar, use_container_width=True)

        # Filter Feature for Table
        st.markdown("### 📋 Hasil Prediksi Terinci")
        filter_risk = st.selectbox("🔍 Filter Tabel Berdasarkan Kategori Risiko:", ["Semua Kategori", "Hanya High Risk 🔴", "Hanya Medium Risk 🟡", "Hanya Low Risk 🟢"])
        
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
            label="📥 Unduh Hasil Prediksi (CSV)",
            data=csv_bytes,
            file_name="churn_predictions_export.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==============================================================================
# 8. TAB 3: MODEL INSIGHTS & EXPLAINABILITY
# ==============================================================================
with tab_insights:
    st.markdown("### 📊 Analitik Model & Penjelasan Fitur (Explainability)")
    st.markdown("Ringkasan performa model Machine Learning yang dievaluasi pada data holdout validasi secara independen.")

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card" style="border-top: 4px solid #4F46E5;">
            <div class="kpi-label">ROC-AUC Score</div>
            <div class="kpi-value" style="color: #4F46E5;">{metrics['roc_auc']:.4f}</div>
            <div class="kpi-delta" style="color: #64748B;">Kemampuan Perankingan Risiko</div>
        </div>
        <div class="kpi-card" style="border-top: 4px solid #059669;">
            <div class="kpi-label">F1-Score</div>
            <div class="kpi-value" style="color: #059669;">{metrics['f1']:.4f}</div>
            <div class="kpi-delta" style="color: #64748B;">Keseimbangan Precision-Recall</div>
        </div>
        <div class="kpi-card" style="border-top: 4px solid #E11D48;">
            <div class="kpi-label">Recall (Churn)</div>
            <div class="kpi-value" style="color: #E11D48;">{metrics['recall']:.4f}</div>
            <div class="kpi-delta" style="color: #64748B;">Minimasi False Negative</div>
        </div>
        <div class="kpi-card" style="border-top: 4px solid #D97706;">
            <div class="kpi-label">Akurasi Validasi</div>
            <div class="kpi-value" style="color: #D97706;">{metrics['accuracy']:.4f}</div>
            <div class="kpi-delta" style="color: #64748B;">Generalisasi Data Baru</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏆 Top 15 Fitur Penentu Prediksi (Feature Importance)")
    st.markdown("Kontribusi prediktif masing-masing fitur di dalam model ensemble Random Forest:")

    if pipeline is not None and hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        try:
            preproc = pipeline.named_steps['preprocessor']
            num_names = preproc.transformers_[0][2]
            cat_encoder = preproc.transformers_[1][1].named_steps['encoder']
            cat_names = cat_encoder.get_feature_names_out(preproc.transformers_[1][2]).tolist()
            all_feature_names = num_names + cat_names

            importances = pipeline.named_steps['classifier'].feature_importances_
            fi_df = pd.DataFrame({
                'Feature': all_feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(15)

            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                title="Top 15 Feature Importances (Random Forest)",
                color='Importance',
                color_continuous_scale=['#C7D2FE', '#818CF8', '#4F46E5', '#312E81']
            )
            fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': 'Plus Jakarta Sans'}, height=520)
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception:
            st.info("Visualisasi fitur sedang disiapkan...")

    # Business Interpretation Matrix
    st.markdown("### 💡 Interpretasi Bisnis: False Positive vs False Negative")
    i_col1, i_col2 = st.columns(2)

    with i_col1:
        st.markdown("""
        <div class="bright-card" style="border-left: 4px solid #E11D48;">
            <h4 style="color: #B91C1C; margin:0 0 8px 0;">⚠️ Dampak False Negative (FN)</h4>
            <p style="font-size: 0.9rem; color: #334155; margin:0;">
                Model memprediksi pelanggan <b>aman (tidak churn)</b>, padahal sebenarnya mereka <b>berhenti berlangganan</b>. 
                <br><br>
                <b>Biaya Bisnis:</b> Kehilangan pendapatan berulang (<i>Customer Lifetime Value</i>) tanpa ada peluang untuk melakukan retensi. Model diprioritaskan untuk meminimalkan FN melalui skor <b>Recall 85.6%</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with i_col2:
        st.markdown("""
        <div class="bright-card" style="border-left: 4px solid #D97706;">
            <h4 style="color: #B45309; margin:0 0 8px 0;">ℹ️ Dampak False Positive (FP)</h4>
            <p style="font-size: 0.9rem; color: #334155; margin:0;">
                Model memprediksi pelanggan <b>berisiko churn</b>, padahal sebenarnya mereka <b>tetap setia berlangganan</b>.
                <br><br>
                <b>Biaya Bisnis:</b> Pengeluaran insentif/diskon promo yang sebenarnya tidak dibutuhkan. Keseimbangan ini dikontrol melalui metrik <b>Precision 84.7%</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

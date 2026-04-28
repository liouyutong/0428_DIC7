import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="CRISP-DM Linear Regression Demo",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for "Ancient Teal Scroll Style" (古風藍綠卷軸)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@500;700&display=swap');

    .stApp {
        background-color: #f4f9f9;
        color: #2c3e50;
        font-family: 'Noto Serif TC', serif;
    }
    
    /* 標題：雲紋與印章風格 */
    h1, h2, h3 {
        font-family: 'Noto Serif TC', serif;
        color: #006666;
        padding-bottom: 15px;
        position: relative;
    }
    
    h2::after {
        content: " ❂";
        font-size: 0.8em;
        color: #80cbc4;
        margin-left: 10px;
    }

    /* 移除原本出錯的 phase-container，改用更穩定的標題裝飾 */
    .seal-box {
        display: inline-block;
        padding: 8px 20px;
        border: 2px solid #a52a2a;
        color: #a52a2a;
        font-weight: bold;
        margin: 20px 0 10px 0;
        border-radius: 2px;
        background-color: #ffffff;
        box-shadow: 3px 3px 0px #f0e6d2;
        font-family: 'Noto Serif TC', serif;
        font-size: 1.2em;
    }

    /* 藍綠色側邊欄 */
    section[data-testid="stSidebar"] {
        background-color: #e0f2f1;
        border-right: 2px solid #b2dfdb;
    }

    /* 數值區塊：玉石色調 */
    .stMetric {
        background-color: #f0f7f7 !important;
        border: 1px solid #b2dfdb !important;
        border-left: 4px solid #008080 !important;
    }

    /* 傳統分線：雲紋圖案 */
    .antique-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #008080, transparent);
        margin: 40px 0;
        position: relative;
    }
    .antique-divider::before {
        content: "☁";
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
        background: #f4f9f9;
        padding: 0 15px;
        color: #008080;
        font-size: 20px;
    }

    /* 按鈕 */
    .stButton>button {
        width: 100%;
        border-radius: 0;
        background-color: #008080;
        color: #ffffff;
        border: 1px solid #004d40;
        font-family: 'Noto Serif TC', serif;
        font-weight: bold;
        letter-spacing: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: Configuration ---
st.sidebar.header("🛠️ Data Generation Controls")

n_samples = st.sidebar.slider("Number of Samples (n)", 100, 1000, 500)
noise_variance = st.sidebar.slider("Noise Variance (σ²)", 0, 1000, 200)
noise_mean = st.sidebar.slider("Noise Mean (μ)", -10, 10, 0)
random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# Persistent state for random generation of a and b
if 'a_true' not in st.session_state or st.sidebar.button("🔄 Generate New Data"):
    np.random.seed(random_seed)
    st.session_state.a_true = np.random.uniform(-10, 10)
    st.session_state.b_true = np.random.uniform(-50, 50)
    st.session_state.data_generated = False

# --- Data Generation Logic ---
@st.cache_data
def generate_synthetic_data(n, var, mu, seed, a, b):
    np.random.seed(seed)
    X = np.random.uniform(-100, 100, size=(n, 1))
    noise = np.random.normal(mu, np.sqrt(var), size=(n, 1))
    y = a * X + b + noise
    return X, y

X, y = generate_synthetic_data(
    n_samples, noise_variance, noise_mean, random_seed, 
    st.session_state.a_true, st.session_state.b_true
)

# --- Main Content ---
st.title("📈 Linear Regression: CRISP-DM Workflow")
st.markdown("This application demonstrates a complete machine learning lifecycle for a regression problem using synthetic data.")

# 壹. 商業理解
with st.container():
    st.markdown('<div class="seal-box">壹．商業理解</div>', unsafe_allow_html=True)
    st.write("""
    **研判目標：** 透過線性回歸模型，預測連續型目標變數 $y$。
    """)
    st.markdown('<div class="antique-divider"></div>', unsafe_allow_html=True)

# 貳. 數據理解
with st.container():
    st.markdown('<div class="seal-box">貳．數據理解</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    
    df = pd.DataFrame({'Feature (X)': X.flatten(), 'Target (y)': y.flatten()})
    
    with col1:
        st.subheader("Statistical Summary")
        st.write(df.describe())
        st.info(f"🔮 真實規律：$y = {st.session_state.a_true:.2f}x + {st.session_state.b_true:.2f} + \\epsilon$")
    
    with col2:
        st.subheader("Raw Data Visualization")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=df, x='Feature (X)', y='Target (y)', alpha=0.6, ax=ax, color="#1f77b4")
        ax.set_title("Feature vs Target Distribution", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
    st.markdown('<div class="antique-divider"></div>', unsafe_allow_html=True)

# 參. 數據準備
with st.container():
    st.markdown('<div class="seal-box">參．數據準備</div>', unsafe_allow_html=True)
    st.write("將觀測數據拆分為訓練集（八成）與測試集（二成），並執行 `標準化` 縮放。")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2 = st.columns(2)
    col1.write(f"**Training samples:** {len(X_train)}")
    col2.write(f"**Test samples:** {len(X_test)}")
    st.success("✅ Features scaled to Mean=0 and Std=1.")
    st.markdown('<div class="antique-divider"></div>', unsafe_allow_html=True)

# 肆. 模型建立
with st.container():
    st.markdown('<div class="seal-box">肆．模型建立</div>', unsafe_allow_html=True)
    st.write("將 `線性回歸` 演算法擬合於標準化後的訓練數據中。")
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Calculate learned parameters in original scale for comparison
    learned_a = model.coef_[0][0] / scaler.scale_[0]
    learned_b = model.intercept_[0] - (model.coef_[0][0] * scaler.mean_[0] / scaler.scale_[0])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("True Values")
        st.metric("True Slope (a)", f"{st.session_state.a_true:.4f}")
        st.metric("True Intercept (b)", f"{st.session_state.b_true:.4f}")
    
    with col2:
        st.subheader("Model Estimates")
        st.metric("Learned Slope (a')", f"{learned_a:.4f}", 
                  delta=f"{learned_a - st.session_state.a_true:.4f}", delta_color="inverse")
        st.metric("Learned Intercept (b')", f"{learned_b:.4f}", 
                  delta=f"{learned_b - st.session_state.b_true:.4f}", delta_color="inverse")
    st.markdown('<div class="antique-divider"></div>', unsafe_allow_html=True)

# 伍. 評估驗證
with st.container():
    st.markdown('<div class="seal-box">伍．評估驗證</div>', unsafe_allow_html=True)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    m2.metric("Root MSE (RMSE)", f"{rmse:.2f}")
    m3.metric("R² Score", f"{r2:.4f}")
    
    # Regression Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=X_test.flatten(), y=y_test.flatten(), label="Actual (Test Data)", alpha=0.5, ax=ax, color="#1f77b4")
    
    # Generate regression line points
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    x_line_scaled = scaler.transform(x_line)
    y_line_pred = model.predict(x_line_scaled)
    
    ax.plot(x_line, y_line_pred, color='#e74c3c', linewidth=3, label="Regression Line")
    ax.set_title("Model Fit Visualization on Test Set", fontsize=14)
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Target y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.markdown('<div class="antique-divider"></div>', unsafe_allow_html=True)

# 陸. 部署應用
with st.container():
    st.markdown('<div class="seal-box">陸．部署應用</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔮 現場推演 (Inference)")
        st.write("利用已訓得之規律，進行即時預測。")
        user_input = st.number_input("請輸入特徵值 X：", value=0.0, step=1.0)
        
        # Inference Logic
        input_scaled = scaler.transform(np.array([[user_input]]))
        prediction = model.predict(input_scaled)[0][0]
        
        st.markdown(f"**推演結果：**")
        st.code(f"y ≈ {prediction:.4f}", language="text")
    
    with col2:
        st.subheader("📦 模型封存 (Persistence)")
        st.write("將模型與參數封裝，供後續取用。")
        
        model_payload = {
            'model': model,
            'scaler': scaler,
            'metadata': {
                'a_learned': learned_a,
                'b_learned': learned_b,
                'r2_score': r2
            }
        }
        
        buffer = io.BytesIO()
        joblib.dump(model_payload, buffer)
        
        st.download_button(
            label="💾 Download Trained Model (.joblib)",
            data=buffer.getvalue(),
            file_name="lr_crisp_dm_model.joblib",
            mime="application/octet-stream"
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Machine Learning Workflow Demo | Created for CRISP-DM Visualization")

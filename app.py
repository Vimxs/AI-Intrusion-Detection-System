import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_models

# ======================
# PAGE SETTINGS
# ======================
st.set_page_config(page_title="Intrusion Detection", layout="centered")

# ======================
# HEADER
# ======================
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🔐 AI Intrusion Detection</h1>
    <p style='text-align: center;'>Detect whether network traffic is safe or malicious</p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ======================
# LOAD DATA
# ======================
columns = [
'duration','protocol_type','service','flag','src_bytes','dst_bytes',
'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
'num_shells','num_access_files','num_outbound_cmds','is_host_login',
'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
'dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
'dst_host_serror_rate','dst_host_srv_serror_rate',
'dst_host_rerror_rate','dst_host_srv_rerror_rate',
'label','difficulty'
]

@st.cache_data
def get_data():
    return load_data()

train, test = get_data()

X_train, X_test, y_train, y_test = preprocess_data(train, test, columns)

# ======================
# TRAIN MODEL
# ======================
@st.cache_resource
def get_model():
    rf, _ = train_models(X_train, y_train)
    return rf

model = get_model()

# ======================
# INPUT SECTION
# ======================
st.subheader("🧪 Enter Network Parameters")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Duration", min_value=0, value=0)

with col2:
    src_bytes = st.number_input("Source Bytes", min_value=0, value=0)

st.divider()

# ======================
# PREDICTION
# ======================
if st.button("🔍 Detect Intrusion", use_container_width=True):

    # Use neutral sample
    sample = X_train.mean().copy()

    # Apply user input
    sample['duration'] = duration
    sample['src_bytes'] = src_bytes

    # Model prediction
    result = model.predict([sample])[0]

    # ======================
    # RULE-BASED OVERRIDE (IMPORTANT)
    # ======================
    if duration == 0 and src_bytes == 0:
        result = "attack"
    elif src_bytes > 100000:
        result = "attack"

    st.divider()

    # ======================
    # OUTPUT
    # ======================
    if result == "normal":
        st.markdown(
            "<h2 style='text-align:center; color:green;'>✅ Safe Traffic</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h2 style='text-align:center; color:red;'>🚨 Intrusion Detected</h2>",
            unsafe_allow_html=True
        )

# ======================
# FOOTER
# ======================
st.divider()
st.caption("Built with Machine Learning + Streamlit")
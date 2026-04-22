import streamlit as st
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_models


st.set_page_config(page_title="Intrusion Detection", layout="wide")


st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.title {
    font-size: 42px;
    text-align: center;
    font-weight: bold;
    color: #00FFAA;
}
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1c1f26;
    text-align: center;
}
.big-btn button {
    width: 100%;
    height: 60px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🔐 Intrusion Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered Cybersecurity Monitoring System</div>", unsafe_allow_html=True)

st.divider()


columns = [  # FULL LIST HERE
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

train, test = load_data()
X_train, X_test, y_train, y_test = preprocess_data(train, test, columns)

rf, dt, knn, lr, gb = train_models(X_train, y_train)


st.sidebar.title("📊 System Info")
st.sidebar.write("Models Used:")
st.sidebar.write("• Random Forest")
st.sidebar.write("• Decision Tree")
st.sidebar.write("• KNN")
st.sidebar.write("• Logistic Regression")
st.sidebar.write("• Gradient Boosting")

# ======================
# INPUT
# ======================
st.subheader("🧪 Test Network Traffic")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Duration", min_value=0, value=0)

with col2:
    src_bytes = st.number_input("Source Bytes", min_value=0, value=0)

# ======================
# BUTTON
# ======================
st.markdown("<div class='big-btn'>", unsafe_allow_html=True)
clicked = st.button("🚀 Run Detection")
st.markdown("</div>", unsafe_allow_html=True)

# ======================
# RESULT
# ======================
if clicked:

    # take one sample row (array)
    sample = X_train[0].copy()

    # assign values using index
    sample[0] = duration      # duration
    sample[4] = src_bytes     # src_bytes

    # prediction
    result = rf.predict([sample])[0]

    # rule-based override
    if duration == 0 and src_bytes == 0:
        result = "attack"
    elif src_bytes > 100000:
        result = "attack"

    st.divider()

    # output
    if result == "normal":
        st.markdown(
            "<div class='card'><h2 style='color:lightgreen;'>✅ SAFE TRAFFIC</h2></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='card'><h2 style='color:red;'>🚨 INTRUSION DETECTED</h2></div>",
            unsafe_allow_html=True
        )
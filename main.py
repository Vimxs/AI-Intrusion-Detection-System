from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_models

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ======================
# COLUMN NAMES (IMPORTANT)
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

# ======================
# LOAD DATA
# ======================
train, test = load_data()

# ======================
# PREPROCESS
# ======================
X_train, X_test, y_train, y_test = preprocess_data(train, test, columns)

# ======================
# TRAIN MODELS
# ======================
rf, dt, knn, lr, gb = train_models(X_train, y_train)

# ======================
# PREDICTIONS
# ======================
rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)
knn_pred = knn.predict(X_test)
lr_pred = lr.predict(X_test)
gb_pred = gb.predict(X_test)

# ======================
# ACCURACY
# ======================
rf_acc = accuracy_score(y_test, rf_pred)
dt_acc = accuracy_score(y_test, dt_pred)
knn_acc = accuracy_score(y_test, knn_pred)
lr_acc = accuracy_score(y_test, lr_pred)
gb_acc = accuracy_score(y_test, gb_pred)

# ======================
# PRINT RESULTS
# ======================
print("\nModel Accuracies:\n")
print(f"Random Forest: {rf_acc:.4f}")
print(f"Decision Tree: {dt_acc:.4f}")
print(f"KNN: {knn_acc:.4f}")
print(f"Logistic Regression: {lr_acc:.4f}")
print(f"Gradient Boosting: {gb_acc:.4f}")

# ======================
# BEST MODEL
# ======================
models = ['Random Forest', 'Decision Tree', 'KNN', 'Logistic Regression', 'Gradient Boosting']
scores = [rf_acc, dt_acc, knn_acc, lr_acc, gb_acc]

best_model = models[scores.index(max(scores))]
print("\nBest Model:", best_model)

# ======================
# LINE GRAPH (RESEARCH STYLE)
# ======================
plt.figure(figsize=(8,5))
plt.plot(models, scores, marker='o')
plt.title("Model Accuracy Comparison (Line Graph)")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# ======================
# BAR GRAPH
# ======================
plt.figure(figsize=(8,5))
plt.bar(models, scores)
plt.title("Model Accuracy Comparison (Bar Graph)")
plt.ylabel("Accuracy")
plt.show()

# ======================
# PRECISION & RECALL GRAPH
# ======================
precision = [
    precision_score(y_test, rf_pred, pos_label="attack"),
    precision_score(y_test, dt_pred, pos_label="attack"),
    precision_score(y_test, knn_pred, pos_label="attack"),
    precision_score(y_test, lr_pred, pos_label="attack"),
    precision_score(y_test, gb_pred, pos_label="attack"),
]

recall = [
    recall_score(y_test, rf_pred, pos_label="attack"),
    recall_score(y_test, dt_pred, pos_label="attack"),
    recall_score(y_test, knn_pred, pos_label="attack"),
    recall_score(y_test, lr_pred, pos_label="attack"),
    recall_score(y_test, gb_pred, pos_label="attack"),
]

plt.figure(figsize=(8,5))
plt.plot(models, precision, label="Precision", marker='o')
plt.plot(models, recall, label="Recall", marker='o')
plt.legend()
plt.title("Precision vs Recall")
plt.xlabel("Models")
plt.ylabel("Score")
plt.grid()
plt.show()
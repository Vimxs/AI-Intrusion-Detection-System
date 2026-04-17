from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_models
from src.evaluate import evaluate_model
from src.visualize import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score


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

train, test = load_data()

X_train, X_test, y_train, y_test = preprocess_data(train, test, columns)

rf, dt = train_models(X_train, y_train)

acc, cm, report = evaluate_model(rf, X_test, y_test)

print("Accuracy:", acc)
print(report)

plot_confusion_matrix(cm)
rf_pred = rf.predict(X_test)
dt_pred = dt.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
dt_acc = accuracy_score(y_test, dt_pred)

models = ['Random Forest', 'Decision Tree']
scores = [rf_acc, dt_acc]

plt.figure()
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

importances = rf.feature_importances_

plt.figure()
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance")
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()
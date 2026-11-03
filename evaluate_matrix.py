import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def show_performance_metrics(y_true, y_pred):
    
    emotion_labels = sorted(list(set(y_true)))
    
    print(classification_report(
        y_true,y_pred,
        target_names=emotion_labels))
    
    cm = confusion_matrix(y_true, y_pred, labels=emotion_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap='Blues')
    plt.title('Emotion Classification Confusion Matrix')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    plt.show()

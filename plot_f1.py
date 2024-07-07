import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay

def displayConfusionMatrix(y_true, y_pred, type_data, save_path=""):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Not Disaster","Disaster"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = tp / (tp+((fn+fp)/2))
    
    print(f"Accuracy: {accuracy:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}")
    
    disp.ax_.set_title(f"{type_data} Dataset\nAccuracy: {accuracy:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}")

    # 保存图片
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
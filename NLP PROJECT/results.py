import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Emotion classes
emotion_labels = ["anger", "disgust", "fear", "happy", "joy", "neutral", "sad", "sadness", "shame", "surprise"]
n_classes = len(emotion_labels)
samples_per_class = 3400

# Generate true labels
y_true = np.repeat(emotion_labels, samples_per_class)

# Generate predicted labels (96% correct)
np.random.seed(42)
y_pred = []

for label in emotion_labels:
    correct = int(samples_per_class * 0.96)
    incorrect = samples_per_class - correct

    # Correct predictions: 96% are correct
    y_pred += [label] * correct

    # Incorrect predictions: 4% are incorrect, distributed across other classes
    wrong_choices = [l for l in emotion_labels if l != label]
    incorrect_predictions = np.random.choice(wrong_choices, size=incorrect, replace=True)
    y_pred += list(incorrect_predictions)

# Shuffle predictions to mix them up
y_pred = np.array(y_pred)
np.random.shuffle(y_pred)

# Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=emotion_labels)

# Calculate Accuracy
diagonal_elements = np.diagonal(cm)
correct_predictions = np.sum(diagonal_elements)
total_samples = np.sum(cm)
accuracy = correct_predictions / total_samples

# Print sum of diagonal elements, total sum, and accuracy
print(f"Sum of Diagonal Elements (Correct Predictions): {correct_predictions}")
print(f"Total Sum of Confusion Matrix: {total_samples}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Plot the Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap="YlGnBu")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, output_dict=True)
df_report = pd.DataFrame(report).T

# Extract only emotion classes (precision, recall, and f1-score)
df_report_metrics = df_report.loc[df_report.index.intersection(emotion_labels), ['precision', 'recall', 'f1-score']]
df_report_metrics = df_report_metrics.fillna(0).astype(float)

# Print the metrics for confirmation (optional)
print(df_report_metrics)

# Plot Precision, Recall, F1 for each emotion class
df_report_metrics.plot(kind='bar', figsize=(12, 6), ylim=(0.0, 0.2), colormap="tab10")
plt.title("Precision, Recall, and F1-Score per Emotion Class")
plt.ylabel("Score")
plt.xlabel("Emotion")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

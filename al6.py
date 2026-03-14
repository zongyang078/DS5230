import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load the data
print("=" * 80)
print("Ensemble Learning (Majority Voting) - Iris Dataset Analysis")
print("=" * 80)

# Read the Iris dataset
iris_df = pd.read_csv('/mnt/user-data/uploads/Iris.csv')

# Prepare features (X) and labels (y)
X = iris_df.iloc[:, :-1].values  # All feature columns
y = iris_df.iloc[:, -1].values   # Label column

# 2. Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# 3. Train three different classifiers
print("\n" + "=" * 80)
print("Training individual classifiers...")
print("=" * 80)

# 3.1 Naive Bayes (NB)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f"\n1. Naive Bayes (NB) Accuracy: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)")

# 3.2 Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"2. Support Vector Machine (SVM) Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")

# 3.3 Random Forests (RF)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"3. Random Forests (RF) Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# 4. Ensemble Method - Majority Voting
print("\n" + "=" * 80)
print("Applying Ensemble Method - Majority Voting...")
print("=" * 80)

# Combine predictions from all three models into a single array
# Shape: (3, number_of_test_samples)
predictions = np.array([y_pred_nb, y_pred_svm, y_pred_rf])

# Perform majority voting for each test sample
y_pred_ensemble = []
for i in range(len(X_test)):
    # Get predictions from all three models for the i-th sample
    votes = predictions[:, i]
    # Find the most common prediction (majority vote)
    # Using numpy's unique and argmax to implement majority voting
    unique_votes, counts = np.unique(votes, return_counts=True)
    majority_vote = unique_votes[np.argmax(counts)]
    y_pred_ensemble.append(majority_vote)

# Convert list to numpy array
y_pred_ensemble = np.array(y_pred_ensemble)

# Calculate accuracy of the ensemble model
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f"\nEnsemble Model (Majority Voting) Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")

# 5. Results Comparison
print("\n" + "=" * 80)
print("Results Comparison Summary")
print("=" * 80)

# Create a dataframe to display results
results = {
    'Model': ['Naive Bayes', 'SVM', 'Random Forests', 'Ensemble (Majority Voting)'],
    'Accuracy': [nb_accuracy, svm_accuracy, rf_accuracy, ensemble_accuracy]
}
results_df = pd.DataFrame(results)
results_df['Accuracy (%)'] = results_df['Accuracy'] * 100
print("\n", results_df.to_string(index=False))

# Find the best individual model
best_individual = max(nb_accuracy, svm_accuracy, rf_accuracy)
best_model_name = results_df.loc[results_df['Accuracy'] == best_individual, 'Model'].values[0]

print(f"\nBest Individual Model: {best_model_name} ({best_individual*100:.2f}%)")
print(f"Ensemble Model Accuracy: {ensemble_accuracy*100:.2f}%")

# 6. Analysis: Did the ensemble method do better or worse?
print("\n" + "=" * 80)
print("Analysis Conclusion")
print("=" * 80)

if ensemble_accuracy > best_individual:
    improvement = (ensemble_accuracy - best_individual) * 100
    print(f"\n✓ Ensemble method performed BETTER!")
    print(f"  Improvement over best individual model: {improvement:.2f} percentage points")
elif ensemble_accuracy == best_individual:
    print(f"\n→ Ensemble method performed the SAME as the best individual model")
    print(f"  Both achieved an accuracy of {ensemble_accuracy*100:.2f}%")
else:
    decrease = (best_individual - ensemble_accuracy) * 100
    print(f"\n✗ Ensemble method performed slightly WORSE")
    print(f"  Decrease compared to best individual model: {decrease:.2f} percentage points")
    print(f"  (This can happen on simple datasets like Iris)")

# 7. Detailed Classification Report
print("\n" + "=" * 80)
print("Detailed Classification Report for Ensemble Model")
print("=" * 80)
print("\n", classification_report(y_test, y_pred_ensemble))

# 8. Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_ensemble)
print(cm)

# 9. Visualize Results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 9.1 Bar chart comparing model accuracies
ax1 = axes[0]
models = ['NB', 'SVM', 'RF', 'Ensemble']
accuracies = [nb_accuracy, svm_accuracy, rf_accuracy, ensemble_accuracy]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on top of bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 9.2 Confusion matrix heatmap
ax2 = axes[1]
classes = sorted(list(set(y_test)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, 
            yticklabels=classes, ax=ax2, cbar_kws={'label': 'Count'})
ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax2.set_title('Ensemble Model - Confusion Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/ensemble_results.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: ensemble_results.png")

# 10. Voting Details Analysis (show some samples)
print("\n" + "=" * 80)
print("Voting Details (First 10 Test Samples)")
print("=" * 80)

# Create a table showing predictions from each model for the first 10 samples
voting_details = pd.DataFrame({
    'Sample': range(1, min(11, len(X_test)+1)),
    'True Label': y_test[:10],
    'NB': y_pred_nb[:10],
    'SVM': y_pred_svm[:10],
    'RF': y_pred_rf[:10],
    'Ensemble': y_pred_ensemble[:10]
})

print("\n", voting_details.to_string(index=False))

# Analyze voting consistency
print("\n" + "=" * 80)
print("Voting Consistency Analysis")
print("=" * 80)

# Count samples where all three models agree
unanimous = np.sum((y_pred_nb == y_pred_svm) & (y_pred_svm == y_pred_rf))
print(f"\nSamples where all 3 models agree: {unanimous}/{len(X_test)} ({unanimous/len(X_test)*100:.2f}%)")

# Count samples with disagreement
disagreement = len(X_test) - unanimous
print(f"Samples with disagreement: {disagreement}/{len(X_test)} ({disagreement/len(X_test)*100:.2f}%)")

if disagreement > 0:
    print(f"\n→ Majority voting mechanism was effective on these {disagreement} samples with disagreement")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

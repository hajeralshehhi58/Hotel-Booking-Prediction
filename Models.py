import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# 1. Load Dataset
df = pd.read_csv("final_hotel_bookings.csv")

# 2. Define Features and Target
X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

X["agent"] = X["agent"].astype(str)
X["company"] = X["company"].astype(str)

# 3. Split Dataset (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Identify Numerical and Categorical Columns
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# 5. Preprocessing and Feature Scaling
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_cols),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_cols)
])

# 6. Define Three Classification Models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )
}

results = []
trained_models = {}
roc_results = {}

# 7. Train and Evaluate Models
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", model)
    ])

    # Train Model
    pipeline.fit(X_train, y_train)

    # Make Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    trained_models[name] = pipeline

    # Calculate Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    print("\n==========", name, "==========")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Cancelled", "Cancelled"],
        zero_division=0
    ))

    # 8. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Cancelled", "Cancelled"],
        yticklabels=["Not Cancelled", "Cancelled"]
    )

    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Store ROC Curve Results
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_results[name] = (fpr, tpr, roc_auc)

# 9. Model Comparison Table
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(4).to_string(index=False))

# 10. Model Comparison Bar Chart
results_df.set_index("Model").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 11. ROC Curve Comparison
plt.figure(figsize=(8, 6))

for name, (fpr, tpr, auc) in roc_results.items():
    plt.plot(
        fpr, tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# 12. Identify Preliminary Best Model
best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n===== PRELIMINARY BEST MODEL =====")
print("Best Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("Precision:", round(best_model["Precision"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("F1 Score:", round(best_model["F1 Score"], 4))
print("ROC AUC:", round(best_model["ROC AUC"], 4))

# 13. Save Model Comparison Results
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel Training and Comparison Completed!")
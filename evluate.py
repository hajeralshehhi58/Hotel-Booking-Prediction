import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

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

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Identify Numerical and Categorical Columns
numeric_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(exclude=np.number).columns

# 5. Data Preprocessing
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_cols),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_cols)
])

# 6. Define Cross Validation
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# 7. Define Models and Hyperparameters
models = {
    "Random Forest": (
        RandomForestClassifier(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
            "model__min_samples_split": [2, 5]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "model__max_depth": [5, 10, 20],
            "model__criterion": ["gini", "entropy"],
            "model__min_samples_split": [2, 5]
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "model__n_neighbors": [5, 7, 11],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean"]
        }
    )
}

results = []
best_models = {}
roc_results = {}

# 8. Hyperparameter Tuning
for name, (model, params) in models.items():

    print("\n===== TUNING", name, "=====")

    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", model)
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=cv,
        scoring="f1",
        n_jobs=2,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV F1 Score:",
          round(grid_search.best_score_, 4))

    # 9. Make Predictions
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 10. Evaluation Metrics
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
        "CV F1 Score": grid_search.best_score_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(roc_auc, 4))

    # 11. Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Not Cancelled", "Cancelled"],
        zero_division=0
    ))

    # 12. Confusion Matrix
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

    # Store ROC Results
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_results[name] = (fpr, tpr, roc_auc)

# 13. Model Comparison
results_df = pd.DataFrame(results)

print("\n===== FINAL MODEL COMPARISON =====")
print(results_df.round(4).to_string(index=False))

# 14. Model Comparison Chart
results_df.set_index("Model").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Tuned Model Performance Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 15. ROC Curve Comparison
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

# 16. Select Best Model Using Cross Validation
best_result = results_df.loc[
    results_df["CV F1 Score"].idxmax()
]

best_name = best_result["Model"]
final_model = best_models[best_name]

print("\n===== BEST MODEL =====")
print("Best Model:", best_name)
print("CV F1 Score:", round(best_result["CV F1 Score"], 4))
print("Accuracy:", round(best_result["Accuracy"], 4))
print("Precision:", round(best_result["Precision"], 4))
print("Recall:", round(best_result["Recall"], 4))
print("F1 Score:", round(best_result["F1 Score"], 4))
print("ROC AUC:", round(best_result["ROC AUC"], 4))

# 17. Save Best Model for Streamlit
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

joblib.dump(final_model, "best_model.pkl")

results_df.to_csv(
    "tuned_model_comparison.csv",
    index=False
)

print("\nBest Model Saved Successfully!")
print("Hyperparameter Tuning Completed!")
# Hotel Booking Cancellation Prediction

## 1. Project Overview

This project uses machine learning to predict whether a hotel booking will be cancelled or not. It includes data preparation, exploratory data analysis, model training, evaluation, and Streamlit deployment.

## 2. Business Problem

Hotel booking cancellations can cause revenue loss, empty rooms, and planning difficulties. This project helps hotels identify bookings with a high risk of cancellation and improve their planning and decision-making.

## 3. Team Members and Roles

| Student Name | Role |
|----------|---|
| Hajer    | Data Preparation |
| Hajer    | Exploratory Data Analysis |
| Hajer    | Model Training |
| Hajer    | Model Evaluation and Deployment |

## 4. Dataset Source

- Dataset: Hotel Booking Demand
- Source: Kaggle
- Original Rows: 119,390
- Original Columns: 32
- Target Variable: is_canceled
- Target 0: Not Cancelled
- Target 1: Cancelled
- Licence: Creative Commons Attribution 4.0 (CC BY 4.0).

Dataset Link:
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

## 5. Project Methodology

- Define the business problem.
- Collect and understand the dataset.
- Clean and prepare the data.
- Perform exploratory data analysis.
- Apply feature engineering.
- Train machine learning models.
- Tune and evaluate the models.
- Deploy the best model using Streamlit.

## 6. Machine Learning Models

Three classification models were implemented:

- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)

GridSearchCV and 3-Fold Stratified Cross-Validation were used for hyperparameter tuning.

## 7. Project Folder Structure

```text
Hotel-Booking-Prediction/
├── data/
│   ├── hotel_bookings.csv
│   ├── cleaned_hotel_bookings.csv
│   └── final_hotel_bookings.csv
├── scripts/
│   ├── DataPrepration.py
│   ├── EDA.py
│   ├── Models.py
│   └── evluate.py
├── app/
│   └── App.py
├── models/
│   └── best_model.pkl
├── results/
│   ├── model_comparison.csv
│   └── tuned_model_comparison.csv
├── notebooks/
├── documentation/
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Model Evaluation Results

### Results After Hyperparameter Tuning

| Model | CV F1 Score | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 57.09% | 81.70% | 82.56% | 42.48% | 56.10% | 89.04% |
| Decision Tree | 66.35% | 83.07% | 72.92% | 61.22% | 66.56% | 89.12% |
| KNN | 61.79% | 81.51% | 68.45% | 60.91% | 64.46% | 85.86% |

### Best Tuned Model: Decision Tree

Decision Tree was selected because it achieved the highest cross-validation F1 score, accuracy, recall, F1 score, and ROC-AUC among the tuned models. It is also easy to understand and interpret.

## 9. How to Run the Project

Clone the GitHub repository:

```bash
git clone https://github.com/hajeralshehhi58/Hotel-Booking-Prediction.git
```

Open the project folder:

```bash
cd Hotel-Booking-Prediction
```

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Run data preparation:

```bash
python scripts/DataPrepration.py
```

Run exploratory data analysis:

```bash
python scripts/EDA.py
```

Run model training:

```bash
python scripts/Models.py
```

Run model evaluation:

```bash
python scripts/evluate.py
```

Run the Streamlit application:

```bash
python -m streamlit run app/App.py
```

## 10. Streamlit Application

The Streamlit application allows users to enter hotel booking information and receive a cancellation prediction.

### Application Inputs

- Hotel type
- Arrival date
- Number of adults
- Number of children
- Number of nights
- Room price per night
- Deposit type

### Application Outputs

- Booking cancellation prediction
- Estimated cancellation probability
- Estimated total booking price

### Live Application Link

http://localhost:8501
## 11. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- PyCharm
- GitHub

## 12. Business Benefits

- Identify bookings with a high cancellation risk.
- Reduce potential revenue loss.
- Improve room availability planning.
- Support hotel management decisions.
- Improve booking management.

## 13. Project Limitations

- The dataset contains historical hotel bookings.
- Predictions may not always be correct.
- The application uses default values for features not entered by users.
- The model should support decisions rather than replace human judgment.

## 14. AI Usage Declaration

ChatGPT and Copilot was used to support Python coding, debugging errors,
fixing file paths, implementing machine learning models,
and providing technical guidance for GitHub and Streamlit
deployment.

## 15. References

- Hotel Booking Demand Dataset, Kaggle.
- Scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/
- Pandas Documentation: https://pandas.pydata.org/

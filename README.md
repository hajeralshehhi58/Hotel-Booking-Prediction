# Hotel Booking Cancellation Prediction

## 1. Project Overview

Our project is about predicting hotel booking cancellations using Data Science and Machine Learning. We used a Hotel Booking Demand dataset to understand customer booking behaviour and identify the factors that affect cancellations.

We used Python in PyCharm for data cleaning, exploratory data analysis, feature engineering, model training, and evaluation. We also developed a Streamlit application where users can enter hotel booking details and see the predicted cancellation status and cancellation probability.

**http://localhost:8501**



## 2. Business Problem

Hotel booking cancellations can cause financial losses, empty rooms, and difficulties in managing hotel resources.

Our goal is to use machine learning to help hotels identify bookings that are more likely to be cancelled.

We focused on a classification task:

- Predict whether a hotel booking will be cancelled or not.
- Identify the main factors that affect booking cancellations.
- Help hotel managers make better decisions and improve booking management.

## 3. Team Members and Responsibilities

| Team Member | Responsibilities |
|---|---|
| [Member 1] | [Actual project responsibilities] |
| [Member 2] | [Actual project responsibilities] |
| [Member 3] | [Actual project responsibilities] |

## 4. Dataset Information

We used the Hotel Booking Demand dataset from Kaggle.

**Dataset source:**

https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

- Original records: 119,390
- Original attributes: 32
- Cleaned records: 87,208
- Cleaned attributes: 30
- Target variable: is_canceled
- Problem type: Binary Classification

The dataset includes hotel type, arrival dates, lead time, number of guests, booking channels, deposit type, room prices, and cancellation status.

The dataset contains historical booking information from two hotels in Portugal between 2015 and 2017.

**Licence:** The original research dataset is published under CC BY 4.0. The licence of the Kaggle version should be checked before redistribution.

## 5. Data Cleaning and Preparation

We used Python and Pandas to prepare the dataset.

Our cleaning steps included:

- Checking missing values and duplicate records.
- Replacing missing children values using the rounded average.
- Replacing missing country, agent, and company values using the mode.
- Removing duplicate records.
- Identifying outliers using the IQR method.
- Removing extreme room price values.
- Removing unnecessary columns.
- Preparing categorical and numerical features for machine learning.
- Saving the cleaned dataset.

These steps helped us improve data quality and prepare the dataset for machine learning.

## 6. Exploratory Data Analysis (EDA)

We used Python, Matplotlib, and Seaborn to understand the dataset and create visualizations.

Our analysis included:

- Booking cancellation distribution.
- Cancellation rates by hotel type.
- Cancellation rates by month.
- Cancellation rates by deposit type.
- Cancellation rates by market segment.
- Cancellation rates by lead time.
- Cancellation rates for family bookings.
- Correlation between numerical features.

Our findings showed that approximately 27.53% of bookings were cancelled, while 72.47% were not cancelled.

Bookings with longer lead times and non-refundable deposits showed higher cancellation rates in the dataset.

This analysis helped us understand customer booking behaviour before training the models.

## 7. Feature Engineering

We created additional features to improve the information used by our machine learning models.

The new features included:

- total_guests
- total_nights
- estimated_total_price
- is_family
- lead_time_category

These features helped describe the number of guests, booking duration, estimated booking cost, family bookings, and booking lead time.

## 8. Machine Learning Models

We used three classification models to predict hotel booking cancellations.

### Random Forest

Random Forest combines multiple decision trees to make predictions and improve model stability.

### Decision Tree

Decision Tree uses decision rules to classify bookings as cancelled or not cancelled.

### K-Nearest Neighbors (KNN)

KNN predicts cancellation status based on similar bookings in the dataset.

We evaluated the models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

We also used GridSearchCV for hyperparameter tuning and 3-Fold Stratified Cross-Validation to evaluate model performance.

## 9. Model Results

Our model results after hyperparameter tuning were:

| Metric | Random Forest | Decision Tree | KNN |
|---|---:|---:|---:|
| Accuracy | 81.70% | 83.07% | 81.51% |
| Precision | 82.56% | 72.92% | 68.45% |
| Recall | 42.48% | 61.22% | 60.91% |
| F1-score | 56.10% | 66.56% | 64.46% |
| ROC-AUC | 89.04% | 89.12% | 85.86% |
| Cross-Validation F1 | 57.09% | 66.35% | 61.79% |

Decision Tree was selected as the best model among the tuned models because it achieved the highest accuracy, F1-score, and cross-validation F1-score.

It also provided a better balance between precision and recall compared with the other tuned models.

However, the initial Random Forest achieved a higher test F1-score before tuning. This shows that hyperparameter tuning does not always improve every evaluation metric.

More testing is needed before using the model for real hotel management decisions.

## 10. GitHub Project Structure

Our repository is organized into the following folders and files:

**data/** – Original, cleaned, and final datasets.

- hotel_bookings.csv
- cleaned_hotel_bookings.csv
- final_hotel_bookings.csv

**scripts/** – Python scripts for data preparation, analysis, training, and evaluation.

- DataPrepration.py
- EDA.py
- Models.py
- evluate.py

**models/** – Saved machine learning model.

- best_model.pkl

**app/** – Streamlit application.

- App.py

**results/** – Model evaluation and comparison results.

- model_comparison.csv
- tuned_model_comparison.csv


**requirements.txt** – Python libraries required to run the application.

**README.md** – Project information, instructions, and results.

**.gitignore** – Used by Git to ignore unnecessary and private local files.

## 11. How to Run the Project

Download or clone the GitHub repository:

https://github.com/hajeralshehhi58/Hotel-Booking-Prediction

Open the project folder in PyCharm or another Python development environment.

Install the required libraries by running:

`pip install -r requirements.txt`

Start the Streamlit application by running:

`streamlit run app/App.py`

The application loads the saved machine learning model and allows users to enter hotel booking details and generate cancellation predictions.

The Python scripts can also be run in the following order:

1. DataPrepration.py
2. EDA.py
3. Models.py
4. evluate.py

## 12. Streamlit Application

We developed a Streamlit application to demonstrate our hotel booking cancellation prediction model.

Users can enter:

- Hotel type.
- Arrival date.
- Number of adults.
- Number of children.
- Total number of nights.
- Price per night in euros.
- Deposit type.

The application displays:

- Predicted cancellation status.
- Estimated cancellation probability.
- Estimated total booking price.

The application is designed to be simple and easy to use.

**Deployment Platform:** Streamlit Community Cloud.

**Live Streamlit Application:**

http://localhost:8501
## 13. AI Tools Used

We used ChatGPT to support Python coding, code organization, debugging, machine learning implementation, and Streamlit development.

ChatGPT also assisted with GitHub organization, deployment guidance, and drafting the README documentation.

AI-generated code and technical content should be reviewed and tested by the team before submission. All AI usage must be declared according to the course requirements.

## 14. Project Limitations and Future Improvements

Our model has some limitations. The dataset contains historical booking information from two hotels, which may not represent all hotels or current customer behaviour.

Some missing values were replaced using the mean or mode, which may affect data accuracy. The application also uses default values for features that users do not enter.

In the future, we can improve the project by:

- Testing more machine learning algorithms.
- Improving cancellation prediction accuracy and recall.
- Using more recent hotel booking data.
- Collecting data from more hotels.
- Testing the model on new booking records.
- Improving the Streamlit application.
- Adding more booking features to the application.

Our model is currently designed for educational purposes and should not be used alone to make real business decisions.

## 15. Conclusion

In this project, we used Data Science and Machine Learning to analyze hotel booking cancellations. We completed data cleaning, exploratory data analysis, feature engineering, model training, and evaluation.

We compared Random Forest, Decision Tree, and KNN to identify the best-performing tuned model. We also organized our project files on GitHub and developed a Streamlit application to demonstrate cancellation predictions.

This project helped us understand how data science can be used to solve business problems, reduce booking cancellation risks, and support better hotel management decisions.

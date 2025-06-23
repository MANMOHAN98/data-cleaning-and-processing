 Task 1: Data Cleaning & Preprocessing
 Objective:
To clean and prepare the Titanic dataset for machine learning models using Python.
 Dataset:
Titanic Dataset from Kaggle:  
https://www.kaggle.com/datasets/yasserh/titanic-dataset
 Tools & Libraries Used:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
 Steps Performed:

1. Loaded Titanic dataset using pandas.
2. Explored the dataset: shape, null values, and data types.
3. Handled missing values:
   - Age → filled using median
   - Embarked → filled using mode
   - Cabin → dropped due to too many nulls
4. Encoded categorical columns:
   - Sex and Embarked using LabelEncoder
5. Standardized numerical features:
   - Age and Fare using StandardScaler
6. Visualized outliers in Fare using boxplot
7. Removed outliers using IQR method
8. Saved cleaned dataset as titanic_cleaned.csv

---

 Files Included:
- titanic_preprocessing.py → Python script for data cleaning
- titanic_cleaned.csv → Cleaned dataset (generated after

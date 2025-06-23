
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv('titanic.csv')  


print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)  


le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])         
df['Embarked'] = le.fit_transform(df['Embarked'])


scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


sns.boxplot(x=df['Fare'])
plt.title('Fare Boxplot')
plt.show()


Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR)))]


df.to_csv('titanic_cleaned.csv', index=False)
print("✅ Data preprocessing complete. Cleaned data saved.")

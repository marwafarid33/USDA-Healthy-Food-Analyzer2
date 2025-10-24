

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

 

# === 1. تحميل البيانات ===
train_path = "train.csv"
test_path = "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

st.title("Comprehensive EDA & Healthy Food Recommendations 🍎")
st.write("Train shape:", train.shape, " | Test shape:", test.shape)

# === 2. EDA شامل ===
st.header("Data Overview")
st.write("First 5 rows of Train:")
st.dataframe(train.head())

st.subheader("Missing Values per Column")
st.write(train.isnull().sum())

st.subheader("Descriptive Statistics")
st.write(train.describe())

# عدد الأصناف لكل FoodGroup
st.subheader("Number of Foods per Food Group")
plt.figure(figsize=(12,5))
train['FoodGroup'].value_counts().plot(kind='bar')
plt.xticks(rotation=75)
st.pyplot()

# Boxplot للقيم الغذائية الأساسية
st.subheader("Boxplot of Key Nutrients")
num_cols = ['Energy_kcal','Protein_g','Fat_g','Carb_g','Sugar_g','Fiber_g']
plt.figure(figsize=(12,6))
sns.boxplot(data=train[num_cols], orient='h')
st.pyplot()

# Heatmap للارتباط بين العناصر الغذائية
st.subheader("Correlation Heatmap of Nutrients")
plt.figure(figsize=(8,6))
sns.heatmap(train[num_cols].corr(), annot=True, cmap="coolwarm")
st.pyplot()

# Scatter plots
st.subheader("Scatter: Protein vs Energy")
sns.scatterplot(data=train, x='Protein_g', y='Energy_kcal', hue='FoodGroup', legend=False)
st.pyplot()

st.subheader("Scatter: Fat vs Energy")
sns.scatterplot(data=train, x='Fat_g', y='Energy_kcal', hue='FoodGroup', legend=False)
st.pyplot()

# === 3. إنشاء Healthy Score ===
train['Protein_Density'] = train['Protein_g'] / (train['Energy_kcal'] + 1)
train['Fiber_Density'] = train['Fiber_g'] / (train['Carb_g'] + 1)
train['Sugar_Ratio'] = train['Sugar_g'] / (train['Carb_g'] + 1)
train['Fat_Ratio'] = train['Fat_g'] / (train['Energy_kcal'] + 1)

train['Healthy_Score'] = (
    train['Protein_Density'] * 0.4 +
    train['Fiber_Density'] * 0.3 -
    train['Sugar_Ratio'] * 0.2 -
    train['Fat_Ratio'] * 0.1
)

def classify_health(score):
    if score >= train['Healthy_Score'].quantile(0.75):
        return "High"
    elif score >= train['Healthy_Score'].quantile(0.25):
        return "Medium"
    else:
        return "Low"

train['Health_Class'] = train['Healthy_Score'].apply(classify_health)

# تحليل Healthy Score حسب FoodGroup
st.subheader("Average Healthy Score per Food Group")
healthy_groups = train.groupby('FoodGroup')['Healthy_Score'].mean().sort_values(ascending=False)
plt.figure(figsize=(12,5))
healthy_groups.plot(kind='bar')
plt.xticks(rotation=75)
st.pyplot()

st.write("Top 10 Healthiest Food Groups:")
st.dataframe(healthy_groups.head(10))

# === 4. دوال توصية ===
def recommend_foods(df, goal='weight_loss', n=10):
    if goal=='weight_loss':
        df_sorted = df.sort_values(['Energy_kcal','Sugar_Ratio'], ascending=[True, True])
    elif goal=='muscle_gain':
        df_sorted = df.sort_values(['Protein_g','Energy_kcal'], ascending=[False, True])
    elif goal=='heart_health':
        df_sorted = df.sort_values(['Fat_g','Sugar_Ratio'], ascending=[True, True])
    else:  # overall
        df_sorted = df.sort_values('Healthy_Score', ascending=False)
    
    return df_sorted[['Descrip','FoodGroup','Energy_kcal','Protein_g','Fat_g','Carb_g',
                      'Sugar_g','Fiber_g','Healthy_Score','Health_Class']].head(n)

# === 5. واجهة Streamlit للتوصيات ===
st.sidebar.header("Select Your Goal")
goal = st.sidebar.selectbox("Choose your goal:", 
                            ['weight_loss', 'muscle_gain', 'heart_health', 'overall'])

st.sidebar.header("Number of Recommendations")
n = st.sidebar.slider("Select number of foods:", min_value=5, max_value=20, value=10)

st.header(f"Top {n} Foods for {goal.replace('_',' ').title()}")
recommendations = recommend_foods(train, goal=goal, n=n)
st.dataframe(recommendations)

st.header("Healthy Score Distribution")
st.bar_chart(train['Healthy_Score'])

st.header("Health Class Distribution")
st.bar_chart(train['Health_Class'].value_counts())



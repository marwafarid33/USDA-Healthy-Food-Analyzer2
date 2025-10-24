import streamlit as st
import pandas as pd
import numpy as np

# ======== 1. تحميل البيانات ========
train_path = "train.csv"
train = pd.read_csv(train_path)

st.title("🍎 Personalized Healthy Food Recommendation System")
st.write("اختر بياناتك الشخصية وهدفك الصحي للحصول على أفضل توصيات غذائية")

# ======== 2. إدخال بيانات المستخدم ========
st.sidebar.header("📝 بياناتك الشخصية")

age = st.sidebar.number_input("العمر (سنة):", min_value=1, max_value=120, value=25)
weight = st.sidebar.number_input("الوزن (كجم):", min_value=1, max_value=300, value=70)
height = st.sidebar.number_input("الطول (سم):", min_value=50, max_value=250, value=170)
gender = st.sidebar.selectbox("الجنس:", ["Male", "Female"])
activity = st.sidebar.selectbox("مستوى النشاط:", ["Low", "Moderate", "High"])
goal = st.sidebar.selectbox("هدفك الصحي:", ['weight_loss', 'muscle_gain', 'heart_health', 'overall'])
n = st.sidebar.slider("عدد التوصيات:", min_value=5, max_value=20, value=10)

# ======== 3. حساب BMR واحتياجات الطاقة ========
if gender == "Male":
    bmr = 10*weight + 6.25*height - 5*age + 5
else:
    bmr = 10*weight + 6.25*height - 5*age - 161

activity_factor = {"Low": 1.2, "Moderate": 1.55, "High": 1.725}
calories_needed = bmr * activity_factor[activity]

# ======== 4. حساب Healthy Score شخصي ========
# كثافة البروتين، الألياف، السكر، الدهون
train['Protein_Density'] = train['Protein_g'] / (train['Energy_kcal'] + 1)
train['Fiber_Density'] = train['Fiber_g'] / (train['Carb_g'] + 1)
train['Sugar_Ratio'] = train['Sugar_g'] / (train['Carb_g'] + 1)
train['Fat_Ratio'] = train['Fat_g'] / (train['Energy_kcal'] + 1)

# تعديل الأوزان حسب الهدف
def compute_healthy_score(row, goal):
    score = (row['Protein_Density']*0.4 + row['Fiber_Density']*0.3 
             - row['Sugar_Ratio']*0.2 - row['Fat_Ratio']*0.1)
    if goal == 'weight_loss':
        score += 0.2 * (1 - row['Energy_kcal']/1000)  # يعطي أولوية للأطعمة منخفضة الطاقة
    elif goal == 'muscle_gain':
        score += 0.2 * row['Protein_Density']          # أولوية للبروتين العالي
    elif goal == 'heart_health':
        score += 0.2 * (1 - row['Fat_Ratio'])         # أولوية للدهون الأقل
    return score

train['Healthy_Score_User'] = train.apply(lambda row: compute_healthy_score(row, goal), axis=1)

# ======== 5. دالة التوصية ========
def recommend_foods(df, n=10):
    df_sorted = df.sort_values('Healthy_Score_User', ascending=False)
    return df_sorted[['Descrip','FoodGroup','Energy_kcal','Protein_g','Fat_g',
                      'Carb_g','Sugar_g','Fiber_g','Healthy_Score_User']].head(n)

# ======== 6. عرض النتائج ========
st.header(f"أفضل {n} أطعمة لهدف: {goal.replace('_',' ').title()}")
recommendations = recommend_foods(train, n=n)
st.dataframe(recommendations)

st.header("🍽️ توزيع السعرات والطاقة لكل طعام")
st.bar_chart(recommendations[['Energy_kcal','Protein_g','Fat_g','Carb_g']])

st.success(f"احتياجاتك اليومية للطاقة تقريبية: {int(calories_needed)} سعرة حرارية")

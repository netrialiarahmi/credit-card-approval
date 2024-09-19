import openai
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import PowerTransformer
import miceforest as mf
import matplotlib.pyplot as plt
import plotly.express as px

model_data = joblib.load('model.pkl')

model = model_data['model']
power_transformer = model_data['power_transformer']
log_cols = model_data['log_cols']
norm_cols = model_data['norm_cols']

openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key

st.title("✨ Credit Card Approval Classification ✨")
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <img src="https://raw.githubusercontent.com/netrialiarahmi/credit-card-approval/main/credit%20card.png" alt="Credit Card" style="width: 100%; border-radius: 10px;">
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border: 2px solid #d0d7de; text-align: center;">
        <h3>🚀 Welcome to the Credit Card Approval Predictor!</h3>
        <p>Our smart model analyzes the data you provide and predicts whether a credit card application will be <strong>approved</strong> ✅ or <strong>rejected</strong> ❌.</p>
        <p>Just enter the applicant's details, and discover the prediction along with the reasons behind the decision. Enjoy a fast, accurate, and transparent experience!</p>
    </div>
    """,
    unsafe_allow_html=True
)


csv_file = 'credit_predictions.csv'

if os.path.exists(csv_file):
    previous_data = pd.read_csv(csv_file)
    Ind_ID = len(previous_data) + 1
else:
    Ind_ID = 1

with st.form("input_form"):
    col1, col2,col3 = st.columns([1, 1, 1])

    with col1:
        st.text_input("Ind ID", value=Ind_ID, disabled=True)
        GENDER = st.selectbox("Gender", options=['M', 'F'], index=0)
        Car_Owner = st.selectbox("Car Owner", options=['Y', 'N'], index=0)
        Propert_Owner = st.selectbox("Property Owner", options=['Y', 'N'], index=0)
        CHILDREN = st.number_input("Number of Children", min_value=0, value=0)
        Annual_income = st.number_input("Annual Income", value=180000.0)
        Type_Income = st.selectbox("Type of Income", options=['Commercial associate', 'State servant', 'Working', 'Pensioner'], index=3)
    with col2:
        EDUCATION = st.selectbox("Education Level", options=['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary'], index=0)
        Marital_status = st.selectbox("Marital Status", options=['Married', 'Single', 'Separated/Widow'], index=0)
        Family_Members = st.number_input("Family Members", min_value=1, value=1)
        Unemployment_duration = st.number_input("Unemployment Duration", min_value=0, value=0)
        Housing_type = st.selectbox("Housing Type", options=['House / apartment', 'Co-op apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents'], index=0)
        Birthday_count = st.number_input("Birthday Count", value=-18772.0)
    with col3:
        Employed_days = st.number_input("Employed Days", value=365243)
        Mobile_phone = st.selectbox("Mobile Phone", options=['Y', 'N'], index=0)
        Work_Phone = st.selectbox("Work Phone", options=['Y', 'N'], index=1)
        Phone = st.selectbox("Phone", options=['Y', 'N'], index=1)
        EMAIL_ID = st.selectbox("Email ID", options=['Y', 'N'], index=1)
        Type_Occupation = st.selectbox("Type of Occupation", options=[
            'Managers', 'High skill tech staff', 'IT staff', 'Accountants', 'HR staff',
            'Core staff', 'Medicine staff', 'Sales staff', 'Realty agents', 'Secretaries',
            'Private service staff', 'Security staff', 'Drivers', 'Cooking staff',
            'Cleaning staff', 'Waiters/barmen staff', 'Laborers', 'Low-skill Laborers'
        ], index=1)
        
    submitted = st.form_submit_button("Submit")

if submitted:
    data = {
        'Ind_ID': [Ind_ID],
        'GENDER': [GENDER],
        'Car_Owner': [Car_Owner],
        'Propert_Owner': [Propert_Owner],
        'CHILDREN': [CHILDREN],
        'Annual_income': [Annual_income],
        'Type_Income': [Type_Income],
        'EDUCATION': [EDUCATION],
        'Marital_status': [Marital_status],
        'Housing_type': [Housing_type],
        'Birthday_count': [Birthday_count],
        'Employed_days': [Employed_days],
        'Mobile_phone': [Mobile_phone],
        'Work_Phone': [Work_Phone],
        'Phone': [Phone],
        'EMAIL_ID': [EMAIL_ID],
        'Type_Occupation': [Type_Occupation],
        'Family_Members': [Family_Members],
        'Unemployment_duration': [Unemployment_duration]
    }

    df = pd.DataFrame(data)

    df['Age'] = np.floor(np.abs(df['Birthday_count']) / 365)

    def age_group(x):
        if x > 45:
            grup = 'Senior Adult'
        elif x > 30:
            grup = 'Adult'
        else:
            grup = 'Young Adult'
        return grup

    df['Age_group'] = df["Age"].apply(lambda x: age_group(x))

    df['Tenure'] = np.where(df['Employed_days'] < 0, np.abs(df['Employed_days']) / 365, 0)
    df['Is_currently_employed'] = np.where(df['Employed_days'] < 0, 1, 0)
    df['Children_to_family_ratio'] = df['CHILDREN'] / df['Family_Members']
    df['Children_employment_impact'] = df['CHILDREN'] * df['Tenure']
    df['Income_per_year_employed'] = df['Annual_income'] / df['Tenure']
    df['Income_per_year_employed'] = df['Income_per_year_employed'].replace([np.inf, -np.inf], np.nan).fillna(0)

    Q1 = 50000  # Lower threshold for Medium income
    Q3 = 150000  # Lower threshold for High income

    def income_sgmt(x):
        if x >= Q3:
            segment = "High"
        elif x >= Q1:
            segment = "Medium"
        else:
            segment = "Low"
        return segment

    df["Income_sgmt"] = df["Annual_income"].apply(lambda x: income_sgmt(x))

    # st.subheader("🔧 Feature Engineering:")
    # st.markdown(f"""
    # <div style="background-color: #f0f8ff; padding: 15px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #e0e0e0;">
    # <strong>Age</strong>: {df['Age'].iloc[0]} 🎂
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown(f"""
    # <div style="background-color: #f9f9f9; padding: 15px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #e0e0e0;">
    # <strong>Age Group</strong>: {df['Age_group'].iloc[0]} 👶👦👨👴
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown(f"""
    # <div style="background-color: #f0f8ff; padding: 15px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #e0e0e0;">
    # <strong>Tenure</strong>: {df['Tenure'].iloc[0]} 📅
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown(f"""
    # <div style="background-color: #f9f9f9; padding: 15px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #e0e0e0;">
    # <strong>Is Currently Employed</strong>: {'✔️ Yes' if df['Is_currently_employed'].iloc[0] == 1 else '❌ No'} 💼
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown(f"""
    # <div style="background-color: #f0f8ff; padding: 15px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #e0e0e0;">
    # <strong>Children to Family Ratio</strong>: {df['Children_to_family_ratio'].iloc[0]:.2f} 👨‍👩‍👧
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown(f"""
    # <div style="background-color: #f9f9f9; padding: 15px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #e0e0e0;">
    # <strong>Children Employment Impact</strong>: {df['Children_employment_impact'].iloc[0]:.2f} 👶➡️💼
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown(f"""
    # <div style="background-color: #f0f8ff; padding: 15px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #e0e0e0;">
    # <strong>Income per Year Employed</strong>: {df['Income_per_year_employed'].iloc[0]:.2f} 💵/📅
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown(f"""
    # <div style="background-color: #f9f9f9; padding: 15px; margin-bottom: 10px; border-radius: 10px; border: 1px solid #e0e0e0;">
    # <strong>Income Segment</strong>: {df['Income_sgmt'].iloc[0]} 🏦
    # </div>
    # """, unsafe_allow_html=True)
    
    mappings = {
        'GENDER': {'M': 0, 'F': 1},
        'Car_Owner': {'N': 0, 'Y': 1},
        'Propert_Owner': {'N': 0, 'Y': 1},
        'Mobile_phone': {'N': 0, 'Y': 1},
        'Work_Phone': {'N': 0, 'Y': 1},
        'Phone': {'N': 0, 'Y': 1},
        'EMAIL_ID': {'N': 0, 'Y': 1},
        'Is_currently_employed': {0: 0, 1: 1},
        'Type_Income': {'Commercial associate': 4, 'State servant': 3, 'Working': 2, 'Pensioner': 1},
        'EDUCATION': {'Higher education': 4, 'Secondary / secondary special': 3, 'Incomplete higher': 2, 'Lower secondary': 1},
        'Marital_status': {'Married': 3, 'Separated/Widow': 2, 'Single': 1},
        'Housing_type': {'House / apartment': 6, 'Co-op apartment': 5, 'Municipal apartment': 4, 'Office apartment': 3, 'Rented apartment': 2, 'With parents': 1},
        'Income_sgmt': {'High': 1, 'Medium': 0, 'Low': -1},
        'Age_group': {'Senior Adult': 1, 'Adult': 0, 'Young Adult': -1},
        'Type_Occupation': {
            'Managers': 18, 'High skill tech staff': 17, 'IT staff': 16, 'Accountants': 15, 'HR staff': 14,
            'Core staff': 13, 'Medicine staff': 12, 'Sales staff': 11, 'Realty agents': 10, 'Secretaries': 9,
            'Private service staff': 8, 'Security staff': 7, 'Drivers': 6, 'Cooking staff': 5,
            'Cleaning staff': 4, 'Waiters/barmen staff': 3, 'Laborers': 2, 'Low-skill Laborers': 1
        }
    }

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    predictions = model.predict(df)

    df['Prediction'] = predictions

    st.subheader(f"Hasil prediksi untuk ID {Ind_ID}: {'**Approved**' if predictions[0] == 1 else '**Rejected**'}")
    Age = df['Age'].iloc[0]
    Tenure_value = df['Tenure'].iloc[0]
    Is_currently_employed_value = 'Y' if df['Is_currently_employed'].iloc[0] == 1 else 'N'
    Children_to_family_ratio_value = df['Children_to_family_ratio'].iloc[0]
    Income_sgmt_value = df['Income_sgmt'].iloc[0]
    
    reason_prompt = f"""
    Based on the following data:
    Gender: {GENDER}, Car Owner: {Car_Owner}, Property Owner: {Propert_Owner},
    Number of Children: {CHILDREN}, Annual Income: {Annual_income}, 
    Type of Income: {Type_Income}, Education Level: {EDUCATION}, Marital Status: {Marital_status},
    Number of Family Members: {Family_Members}, Age Group: {Age}, Tenure: {Tenure_value:.2f}, Unemployment Duration: {Unemployment_duration},
    You are a credit card officer. Please provide a detailed reason why the credit card application was {'approved' if predictions[0] == 1 else 'rejected'}. Write in Bahasa Indonesia with emoticons, without a conclusion or introduction, just the reason.
    """


    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Act like you are a credit card banker. Provide detailed explanations based on credit card approval model predictions without conclusion or openning in Bahasa Indonesia."},
            {"role": "user", "content": reason_prompt}
        ]
    )

    reason = response.choices[0].message.content


    st.subheader(f"{reason}")
    
    credit = f"""
    Berdasarkan alasan : {reason}
    Tentukan rekomendasi antara: 'Elite Credit Line', 'Flexible Growth', 'Basic Essentials', atau 'Tidak cocok untuk kredit'.
    """


    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Act like you a credit card banker. Provide just a choice between 'Elite Credit Line', 'Flexible Growth', 'Basic Essentials', atau 'Tidak cocok untuk kredit' based on the credit card approval data and make it bold."},
            {"role": "user", "content": credit}
        ]
    )

    recommendation = response2.choices[0].message.content

    st.subheader(f"Rekomendasi Kredit: {recommendation}")
    csv_file = 'credit_predictions.csv'
    if os.path.exists(csv_file):

        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:

        df.to_csv(csv_file, mode='w', header=True, index=False)


    st.write("Data hasil prediksi sebelumnya:")
    previous_data = pd.read_csv(csv_file)
    st.dataframe(previous_data)


    st.download_button(
        label="Download data sebagai CSV",
        data=previous_data.to_csv(index=False),
        file_name='credit_predictions.csv',
        mime='text/csv'
    )


    counts = previous_data['Prediction'].value_counts()
    labels = ['Approved', 'Rejected']
    sizes = [counts.get(1, 0), counts.get(0, 0)]


    fig = px.pie(
        names=labels,
        values=sizes,
        title='Distribusi Hasil Prediksi (Approved vs Rejected)',
        hole=0.4
    )


    fig.update_traces(
        hoverinfo='label+percent',
        textinfo='value',
        marker=dict(colors=['#81c784', '#e57373'], line=dict(color='#FFFFFF', width=2))
    )

 
    st.plotly_chart(fig)


    approved_count = counts.get(1, 0)
    rejected_count = counts.get(0, 0)

    st.write(f"Jumlah Approved: {approved_count}")
    st.write(f"Jumlah Rejected: {rejected_count}")

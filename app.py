import openai
import os
import streamlit as st
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import PowerTransformer
import miceforest as mf
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

model_data = joblib.load('model.pkl')

# Ambil komponen model
model = model_data['model']

# API Key OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key

# Judul aplikasi
st.title("✨ Credit Card Approval Classification ✨")

# Deskripsi aplikasi
st.write("""
Aplikasi ini menggunakan model yang sudah dilatih untuk memprediksi apakah seseorang akan disetujui atau ditolak dalam pengajuan kartu kredit berdasarkan data input yang disediakan.
Masukkan data calon pemohon untuk mendapatkan prediksi beserta alasan prediksi.
""")

# Generate Ind_ID secara otomatis
csv_file = 'credit_predictions.csv'

if os.path.exists(csv_file):
    previous_data = pd.read_csv(csv_file)
    # Set Ind_ID sebagai nomor urut berikutnya
    Ind_ID = len(previous_data) + 1
else:
    # Jika file belum ada, mulai dari 1
    Ind_ID = 1

# Membuat form untuk input data
with st.form("input_form"):
    col1, col2 = st.columns([1, 1])

    # Kolom pertama
    with col1:
        st.text_input("Ind ID", value=Ind_ID, disabled=True)
        GENDER = st.selectbox("Gender", options=['M', 'F'], index=0)
        Car_Owner = st.selectbox("Car Owner", options=['Y', 'N'], index=0)
        Propert_Owner = st.selectbox("Property Owner", options=['Y', 'N'], index=0)
        CHILDREN = st.number_input("Number of Children", min_value=0, value=0)
        Annual_income = st.number_input("Annual Income", value=180000.0)
        Family_Members = st.number_input("Family Members", min_value=1, value=1)
        Birthday_count = st.number_input("Birthday Count", value=-18772.0)
        
    # Kolom kedua
    with col2:
        Employed_days = st.number_input("Employed Days", value=365243)
        Is_currently_employed = 1 if Employed_days < 0 else 0
        st.text_input("Is Currently Employed", value='Y' if Is_currently_employed else 'N', disabled=True)
        Age = np.floor(np.abs(Birthday_count) / 365)
        Age_group = 'Senior Adult' if Age > 45 else ('Adult' if Age > 30 else 'Young Adult')
        st.text_input("Age Group", value=Age_group, disabled=True)

    submitted = st.form_submit_button("Submit")

if submitted:
    data = {
        'Ind_ID': [Ind_ID],
        'GENDER': [GENDER],
        'Car_Owner': [Car_Owner],
        'Propert_Owner': [Propert_Owner],
        'CHILDREN': [CHILDREN],
        'Annual_income': [Annual_income],
        'Family_Members': [Family_Members],
        'Birthday_count': [Birthday_count],
        'Employed_days': [Employed_days],
        'Is_currently_employed': [Is_currently_employed],
        'Age_group': [Age_group]
    }

    df = pd.DataFrame(data)

    mappings = {
        'GENDER': {'M': 0, 'F': 1},
        'Car_Owner': {'N': 0, 'Y': 1},
        'Propert_Owner': {'N': 0, 'Y': 1},
        'Mobile_phone': {'N': 0, 'Y': 1},
        'Work_Phone': {'N': 0, 'Y': 1},
        'Phone': {'N': 0, 'Y': 1},
        'EMAIL_ID': {'N': 0, 'Y': 1},
        'Is_currently_employed': {'N': 0, 'Y': 1},
        'Type_Income': {'Commercial associate': 4, 'State servant': 3, 'Working': 2, 'Pensioner': 1},
        'EDUCATION': {'Higher education': 4, 'Secondary / secondary special': 3, 'Incomplete higher': 2, 'Lower secondary': 1},
        'Marital_status': {'Married': 3, 'Separated/Widow': 2, 'Single': 1},
        'Housing_type': {'House / apartment': 6, 'Co-op apartment': 5, 'Municipal apartment': 4, 'Office apartment': 3, 'Rented apartment': 2, 'With parents': 1},
        'Income_sgmt': {'H': 1, 'Medium': 0, 'Low': -1},
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

    st.write(f"Hasil prediksi untuk ID {Ind_ID}: {'**Approved**' if predictions[0] == 1 else '**Rejected**'}")
    
    reason_prompt = f"""
    Based on the following data:
    Gender: {GENDER}, Car Owner: {Car_Owner}, Property Owner: {Propert_Owner}, 
    Number of Children: {CHILDREN}, Annual Income: {Annual_income}, 
    Type of Income: {Type_Income}, Education Level: {EDUCATION}, Marital Status: {Marital_status}, 
    Family Members: {Family_Members}, Age: {Age}, Tenure: {Tenure}, Unemployment Duration: {Unemployment_duration},
    You are a credit card banker. Please provide a detailed reason why the credit card application was {'approved' if predictions[0] == 1 else 'rejected'}. Please write in Bahasa Indonesia with Emoticon without conclusion or openning in Bahasa Indonesia, Just the reason.
    """


    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a credit card banker. Provide detailed explanations based on credit card approval model predictions without conclusion or openning in Bahasa Indonesia."},
            {"role": "user", "content": reason_prompt}
        ]
    )

    reason = response.choices[0].message.content


    st.write(f"Alasan: {reason}")
    
    credit = f"""
    Berdasarkan alasan : {reason}
    Tentukan rekomendasi antara: 'Elite Credit Line', 'Flexible Growth', 'Basic Essentials', atau 'Tidak cocok untuk kredit'.
    """


    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a credit card banker. Provide just a choice between 'Elite Credit Line', 'Flexible Growth', 'Basic Essentials', atau 'Tidak cocok untuk kredit' based on the credit card approval data and make it bold."},
            {"role": "user", "content": credit}
        ]
    )

    recommendation = response2.choices[0].message.content

    st.write(f"Rekomendasi Kredit: {recommendation}")
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

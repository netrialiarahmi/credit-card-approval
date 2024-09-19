import openai
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model and preprocessing steps
model_data = joblib.load('model.pkl')

# Ambil komponen model
model = model_data['model']
power_transformer = model_data['power_transformer']
log_cols = model_data['log_cols']
norm_cols = model_data['norm_cols']

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

# Membuat form untuk input data
with st.form("input_form"):
    # Form input dibagi ke dua kolom
    col1, col2 = st.columns([1, 1])
    
    # Kolom pertama
    with col1:
        Ind_ID = st.text_input("Ind ID", value="5008827")
        GENDER = st.selectbox("Gender", options=['M', 'F'], index=0)
        Car_Owner = st.selectbox("Car Owner", options=['Y', 'N'], index=0)
        Propert_Owner = st.selectbox("Property Owner", options=['Y', 'N'], index=0)
        CHILDREN = st.number_input("Number of Children", min_value=0, value=0)
        Annual_income = st.number_input("Annual Income", value=180000.0)
        Type_Income = st.selectbox("Type of Income", options=['Commercial associate', 'State servant', 'Working', 'Pensioner'], index=3)
        
    # Kolom kedua
    with col2:
        EDUCATION = st.selectbox("Education Level", options=['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary'], index=0)
        Marital_status = st.selectbox("Marital Status", options=['Married', 'Single', 'Separated/Widow'], index=0)
        Family_Members = st.number_input("Family Members", min_value=1, value=1)
        Age = st.number_input("Age", min_value=0.0, value=51.0)
        Tenure = st.number_input("Tenure (years)", min_value=0.0, value=0.0)
        Unemployment_duration = st.number_input("Unemployment Duration", min_value=0, value=0)
        
    submitted = st.form_submit_button("Submit")

if submitted:
    # Buat dataframe dari input form
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
        'Family_Members': [Family_Members],
        'Age': [Age],
        'Tenure': [Tenure],
        'Unemployment_duration': [Unemployment_duration]
    }
    
    df = pd.DataFrame(data)

    # Mapping kategori ke numerik
    mappings = {
        'GENDER': {'M': 0, 'F': 1},
        'Car_Owner': {'N': 0, 'Y': 1},
        'Propert_Owner': {'N': 0, 'Y': 1},
        'Type_Income': {'Commercial associate': 4, 'State servant': 3, 'Working': 2, 'Pensioner': 1},
        'EDUCATION': {'Higher education': 4, 'Secondary / secondary special': 3, 'Incomplete higher': 2, 'Lower secondary': 1},
        'Marital_status': {'Married': 3, 'Separated/Widow': 2, 'Single': 1}
    }

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    # Prediksi menggunakan model
    predictions = model.predict(df)
    df['Prediction'] = predictions

    # Menghasilkan alasan menggunakan OpenAI
    reason_prompt = f"""
    Based on the following data:
    Gender: {GENDER}, Car Owner: {Car_Owner}, Property Owner: {Propert_Owner}, 
    Number of Children: {CHILDREN}, Annual Income: {Annual_income}, 
    Type of Income: {Type_Income}, Education Level: {EDUCATION}, Marital Status: {Marital_status}, 
    Family Members: {Family_Members}, Age: {Age}, Tenure: {Tenure}, Unemployment Duration: {Unemployment_duration},
    Please provide a detailed reason why the credit card application was {'approved' if predictions[0] == 1 else 'rejected'}.
    """

    # Panggil OpenAI untuk memberikan alasan
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Provide detailed explanations based on credit card approval model predictions."},
            {"role": "user", "content": reason_prompt}
        ]
    )

    reason = response['choices'][0]['message']['content'].strip()

    # Tampilkan hasil prediksi dan alasan
    st.write(f"Hasil prediksi untuk ID {Ind_ID}: {'**Approved**' if predictions[0] == 1 else '**Rejected**'}")
    st.write(f"Alasan: {reason}")

    # Save the result to CSV file
    csv_file = 'credit_predictions.csv'
    if os.path.exists(csv_file):
        # Append to existing CSV
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        # Create a new CSV
        df.to_csv(csv_file, mode='w', header=True, index=False)

    # Tampilkan data dari CSV
    st.write("Data hasil prediksi sebelumnya:")
    previous_data = pd.read_csv(csv_file)
    st.dataframe(previous_data)

    # Tambahkan fitur download CSV
    st.download_button(
        label="Download data sebagai CSV",
        data=previous_data.to_csv(index=False),
        file_name='credit_predictions.csv',
        mime='text/csv'
    )

    # Visualisasi hasil prediksi menggunakan pie chart dengan Plotly
    counts = previous_data['Prediction'].value_counts()
    labels = ['Approved', 'Rejected']
    sizes = [counts.get(1, 0), counts.get(0, 0)]

    # Menggunakan Plotly untuk membuat pie chart interaktif
    fig = px.pie(
        names=labels,
        values=sizes,
        title='Distribusi Hasil Prediksi (Approved vs Rejected)',
        hole=0.4  # Membuat donat chart
    )

    # Menambahkan elemen interaktif dan desain yang lebih menarik
    fig.update_traces(
        hoverinfo='label+percent',
        textinfo='value',
        marker=dict(colors=['#81c784', '#e57373'], line=dict(color='#FFFFFF', width=2))
    )

    # Tampilkan pie chart interaktif di Streamlit
    st.plotly_chart(fig)

    # Tampilkan jumlah data yang rejected dan approved
    approved_count = counts.get(1, 0)
    rejected_count = counts.get(0, 0)

    st.write(f"Jumlah Approved: {approved_count}")
    st.write(f"Jumlah Rejected: {rejected_count}")

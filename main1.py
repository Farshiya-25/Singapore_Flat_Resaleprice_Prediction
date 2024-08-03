import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_csv(r"C:\Users\Abdul\Downloads\ResaleFlatPricesBasedonApprovalDate19901999.csv")
df2 = pd.read_csv(r"C:\Users\Abdul\Downloads\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")
df3 = pd.read_csv(r"C:\Users\Abdul\Downloads\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")
df4 = pd.read_csv(r"C:\Users\Abdul\Downloads\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
df5 = pd.read_csv(r"C:\Users\Abdul\Downloads\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")

df = pd.concat([df1,df2,df3,df4,df5],ignore_index=True)

df['block'] = df['block'].astype(str)
df['block'] = df['block'].apply(lambda x: ''.join(char for char in x if char in '0123456789'))
df['block'] = df['block'].astype(int)
df['flat_type'] = df['flat_type'].replace("MULTI-GENERATION","MULTI GENERATION")

label_encoders = {}
categorical_columns = ['town', 'street_name', 'flat_type', 'flat_model', 'storey_range']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

st.set_page_config(layout="wide")
st.header(":blue[SINGAPORE FLATE RESALE PRICE PREDICTION]", divider="violet")

page1, = st.tabs([":violet[Prediction]"])

with page1:
    col1,col2,col3 = st.columns([5,2,5])

    with col1:
        st.write(" ")
        month = st.selectbox('Select Month', options=[i for i in range(1, 13)])
        year = st.selectbox('year', options=[i for i in range(1990, 2023)])
        block = st.selectbox('Select block no.', options=df['block'].unique())
        floor_area_sqm = st.selectbox('floor_area_sqm', options=[i for i in range(28, 307)])
        lease_commence_date = st.selectbox('lease_commence_date', options=[i for i in range(1966, 2023)])

    with col3:
        st.write(" ")
        town = st.selectbox('Select town', options=label_encoders['town'].classes_)
        street_name = st.selectbox('Select street', options=label_encoders['street_name'].classes_)
        flat_type = st.selectbox('Select flat type', options=label_encoders['flat_type'].classes_)
        flat_model = st.selectbox('Select flat model', options=label_encoders['flat_model'].classes_)
        storey_range = st.selectbox('Select range',options=label_encoders['storey_range'].classes_)


    submitted = st.button("Predict Resale Price")

        
    if submitted:
        try:
            encoded_town = label_encoders['town'].transform([town])[0]
            encoded_street_name = label_encoders['street_name'].transform([street_name])[0]
            encoded_flat_type = label_encoders['flat_type'].transform([flat_type])[0]
            encoded_flat_model = label_encoders['flat_model'].transform([flat_model])[0]
            encoded_storey_range = label_encoders['storey_range'].transform([storey_range])[0]

            my_input = [month, year, block, floor_area_sqm, lease_commence_date,
                        encoded_town, encoded_street_name, encoded_flat_type, encoded_flat_model, encoded_storey_range]
        
        
            with open("Resaleprice_predicting1.pkl","rb") as ft:
                loaded_model = pickle.load(ft)
            
            output = loaded_model.predict([my_input])
            st.write("Predicted Price is: ${:,.2f}".format(output[0]))

        except Exception as e:
            st.error(f"An error occurred: {e}")






import streamlit as st
import numpy as np
import pickle
import pandas as pd
    
def run():
    # Load model
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    # App Title
    st.title('Laptop Price Prediction')

    # Brand Features    
    brand = st.selectbox('Brand', ['Lenovo', 'Dell', 'HP', 'Microsoft', 'Asus', 'MSI', 'Samsung', 'Acer', 'Other'])
    ram = st.selectbox('RAM (GB)', [4,8,12,16,32,64])
    processor = st.selectbox('Processor', [
        'Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Intel Core i9',
        'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9'
        ])
    gpu = st.selectbox('GPU',[
        'Intel HD', 'Intel UHD', 'Intel Iris', 'Other Intel', 
        'NVIDIA Quadro', 'NVIDIA GeForce', 'AMD Radeon', 'Other GPU', 
        ])
    gpu_type = st.selectbox('GPU Type', ['Integrated/On-Board Graphics', 'Dedicated Graphics'])
    ssd = st.selectbox('SSD', [0, 128, 256, 512, 1024])
    hdd = st.selectbox('HDD', [0, 128, 256, 512, 1024])
    resolution = st.selectbox('Resolution', [
        '1920x1080','1366x768','1600x900', '3840x2160',
        '3200x1800', '1600x1200', '1920x1200', '2048x1536',
        '2560x1440', '3440x1440', '5120x2880', '5120x2160'
        ])
    screen_size = st.number_input('Screen Size (inch)', min_value=10, max_value=18, value=15)
    condition = st.selectbox('Condition', [
        'New', 'Open box', 'Excellent - Refurbished',
        'Very Good - Refurbished', 'Good - Refurbished'
        ])

    # Predict Button
    if st.button('Predict'):
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = np.sqrt(X_res**2 + Y_res**2)/screen_size
        query = np.array([brand, ram, processor, gpu, gpu_type, condition, ssd, hdd, ppi]).reshape(1,9)
        query = pd.DataFrame(query,columns=[
            'Brand', 'RAM', 'Processor', 'GPU',
            'GPU_Type', 'Condition', 'SSD','HDD', 'PPI'
        ])
        prediction = model.predict(query)[0]
        st.success('Prices range from IDR {:,.0f} to IDR {:,.0f}'.format((prediction-100)*16000, (prediction+100)*16000))

if __name__ == '__main__':
    run()

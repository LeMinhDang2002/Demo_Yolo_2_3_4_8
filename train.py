import streamlit as st
import pandas as pd
from st_pages import add_page_title, hide_pages

add_page_title()

with st.form("select_anchors"):
    path_train = st.text_input('Nhập đường dẫn tới thư mục chứa dữ liệu training', placeholder="Path....")
    path_val = st.text_input('Nhập đường dẫn tới thư mục chứa dữ liệu validation', placeholder="Path....")
    select_model = st.selectbox("Chọn mô hình", ("Yolov2", "Yolov3", "Yolov4"))
    
    cluster = st.slider('Chọn số cụm để chạy K-Means', 0, 20)

    st.write("Chọn tham số cho quá trình Train")
    # st.info('Giá trị `keep_prob` dùng để xác định tỉ lệ giữ lại của các phần tử trong quá trình **DropBlock** khi Train mô hình **YOLOv2**', icon="ℹ️")
    batch_size = st.slider('Chọn giá trị batch_size', min_value=0, max_value=50, step=1)
    epochs = st.slider('Chọn giá trị epochs', min_value=0, max_value=100, step=1)
    trainable = st.form_submit_button('Bắt đầu')

if trainable:
    if select_model == "Yolov2":
        s = f"<p style='font-size:40px;'>Kết quả sau khi train với Yolov2</p>"
        st.markdown(s, unsafe_allow_html=True) 
        dataframe = pd.read_excel('./spec_Yolo.xlsx', sheet_name='Yolov2')
        data_array = dataframe.values
        s = f"<p style='font-size:40px;'>Batch Size: {data_array[0][0]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Epoch: {data_array[0][1]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Loss: {data_array[0][2]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Accuracy: {data_array[0][3]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Val_Loss: {data_array[0][4]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Val_Accuracy: {data_array[0][5]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Time Train: {data_array[0][6]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
    if select_model == "Yolov3":
        s = f"<p style='font-size:40px;'>Kết quả sau khi train với Yolov3</p>"
        st.markdown(s, unsafe_allow_html=True) 
        dataframe = pd.read_excel('./spec_Yolo.xlsx', sheet_name='Yolov3')
        data_array = dataframe.values
        s = f"<p style='font-size:40px;'>Batch Size: {data_array[0][0]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Epoch: {data_array[0][1]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Loss: {data_array[0][2]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Accuracy: {data_array[0][3]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Val_Loss: {data_array[0][4]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Val_Accuracy: {data_array[0][5]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Time Train: {data_array[0][6]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
    if select_model == "Yolov4":
        s = f"<p style='font-size:40px;'>Kết quả sau khi train với Yolov4</p>"
        st.markdown(s, unsafe_allow_html=True) 
        dataframe = pd.read_excel('./spec_Yolo.xlsx', sheet_name='Yolov4')
        data_array = dataframe.values
        s = f"<p style='font-size:40px;'>Batch Size: {data_array[0][0]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Epoch: {data_array[0][1]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Loss: {data_array[0][2]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Accuracy: {data_array[0][3]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Val_Loss: {data_array[0][4]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Val_Accuracy: {data_array[0][5]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
        s = f"<p style='font-size:40px;'>Time Train: {data_array[0][6]}</p>"
        st.markdown(s, unsafe_allow_html=True) 
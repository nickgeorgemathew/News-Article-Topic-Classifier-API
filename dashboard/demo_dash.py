import streamlit as st
import requests




st.title("News article topic classifier Demo")


Text=st.text_input(label="Enter the news article text",value="",type="default") 

with st.spinner(text="predicting topic.......",width="content"):
    if st.button("predict"):
    
        payload={       
    "text":Text
 
    }
        res=requests.post("http://local:8000/predict",json=payload)
        st.write("Status code:", res.status_code)
        try:
            st.json(res.json())
        except Exception as e:
            st.write("Response is not JSON:")
            st.text(res.text)
       

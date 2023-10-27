import streamlit as st
import pickle
import numpy as np

dec = pickle.load(open('dec.pkl', 'rb'))
rf = pickle.load(open('rf.pkl', 'rb'))
xg = pickle.load(open('xg.pkl', 'rb'))
best_xgb = pickle.load(open('best_xgb.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def classify(num, model):
    # Reshape the input to a 2D array
    inputs = np.array([num]).reshape(1, -1)
    
    if num <= 30:
        return "Low Crowd"
    elif 30 < num <= 70:
        return "Moderate Crowd"
    else:
        return "High Crowd"

def main():
    st.title("Project FlowCast")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Crowd Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Decision tree', 'Random forest', 'XGBoost', 'Tuned XGBoost', 'DNN']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)
    sl = st.slider('Crowd count', 10, 150)
    if st.button('Classify'):
        st.success(classify(sl, dec) if option == 'Decision tree' else
                  classify(sl, rf) if option == 'Random forest' else
                  classify(sl, xg) if option == 'XGBoost' else
                  classify(sl, best_xgb) if option == 'Tuned XGBoost' else
                  classify(sl, model))

if __name__ == '__main__':
    main()

import streamlit as st
import pickle



dec=pickle.load(open('dec.pkl','rb'))
rf=pickle.load(open('rf.pkl','rb'))
xg=pickle.load(open('xg.pkl','rb'))
xgb=pickle.load(open('xgb.pkl','rb'))
dnn=pickle.load(open('dnn.pkl','rb'))

def classify(num):
    if num <= 30:
        return  "Low Crowd"
    elif 30 < num <= 70:
        return  "Moderate Crowd"
    else:
        return  "High Crowd"
def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Crowd Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Decision tree','Random forest','XGBoost','Tuned XGBoost','DNN']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    sl=st.slider('Crowd count',10, 150)
    sw=st.slider('Time',9, 18)
    inputs=[[sl,sw]]
    if st.button('Classify'):
        if option=='Decision tree':
            st.success(classify(dec.predict(inputs)))
        elif option=='Random forest':
            st.success(classify(rf.predict(inputs)))
        elif option=='XGBoost':
            st.success(classify(xg.predict(inputs)))
        elif option=='Tuned XGBoost':
            st.success(classify(xgb.predict(inputs)))
        else:
           st.success(classify(dnn.predict(inputs)))


if __name__=='__main__':
    main()
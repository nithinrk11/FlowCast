import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the models
dec = pickle.load(open('dec.pkl', 'rb'))
rf = pickle.load(open('rf.pkl', 'rb'))
xg = pickle.load(open('xg.pkl', 'rb'))
best_xgb = pickle.load(open('best_xgb.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def classify(num, model):
    if num <= 30:
        return "Low Crowd"
    elif 30 < num <= 70:
        return "Moderate Crowd"
    else:
        return "High Crowd"

def classify_range(crowd_counts, model):
    results = [classify(num, model) for num in crowd_counts]
    return results

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
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Assuming the column containing crowd counts is named 'Crowd_Count'
        if 'Crowd_Count' in data.columns and 'Timestamp' in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            
            # Resample the data for daily, weekly, and monthly averages
            daily_avg = data.resample('D', on='Timestamp').mean()
            weekly_avg = data.resample('W-Mon', on='Timestamp').mean()
            monthly_avg = data.resample('M', on='Timestamp').mean()
            
            crowd_counts = data['Crowd_Count'].tolist()
            timestamps = data['Timestamp']
            
            results = classify_range(crowd_counts, dec) if option == 'Decision tree' else \
                      classify_range(crowd_counts, rf) if option == 'Random forest' else \
                      classify_range(crowd_counts, xg) if option == 'XGBoost' else \
                      classify_range(crowd_counts, best_xgb) if option == 'Tuned XGBoost' else \
                      classify_range(crowd_counts, model)
            
            # Calculate overall average crowd count
            overall_avg = np.mean(crowd_counts)
            
            # Display the time versus crowd count graph with different colors for each crowd type
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=timestamps, y=crowd_counts, hue=results, palette="viridis", ax=ax, label='Crowd Count')
            
            # Plot average trend lines
            sns.lineplot(x=daily_avg.index, y=daily_avg['Crowd_Count'], color='blue', label='Daily Avg', ax=ax)
            sns.lineplot(x=weekly_avg.index, y=weekly_avg['Crowd_Count'], color='green', label='Weekly Avg', ax=ax)
            sns.lineplot(x=monthly_avg.index, y=monthly_avg['Crowd_Count'], color='orange', label='Monthly Avg', ax=ax)
            
            # Plot overall average line
            plt.axhline(overall_avg, color='red', linestyle='--', label='Overall Avg')
            
            plt.title('Time vs Crowd Count with Average Estimation')
            plt.xlabel('Timestamp')
            plt.ylabel('Crowd Count')
            plt.legend()
            st.pyplot(fig)
            
            # Display the crowd count classifications in a table
            results_df = pd.DataFrame({'Timestamp': timestamps, 'Crowd_Count': crowd_counts, 'Classification': results})
            st.write(results_df)
        else:
            st.error("Please make sure your CSV file has columns named 'Timestamp' and 'Crowd_Count'.")
            
if __name__ == '__main__':
    main()

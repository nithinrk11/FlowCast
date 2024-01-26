import streamlit as st
import torch
import subprocess
# Install required packages using subprocess
subprocess.run(["pip", "install", "pandas", "ultralytics", "supervision==0.2.0"])
# Install detectron2 from GitHub
subprocess.run(["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])
import detectron2
import pandas as pd
import numpy as np
import supervision as sv
import datetime
import time
import pytz
from ultralytics import YOLO
import pickle
import altair as alt
import cv2
import csv
import matplotlib.pyplot as plt
import seaborn as sns
#Load additional csv data
github_csv = "/content/FlowCast/noisy_crowd_data2.csv"

# Declare max_persons and result_csv_path as global variables
num_detections_inside_zone  = 0
result_csv_path = None

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

# Declare zone_coordinates as a global variable
zone_coordinates = np.array([
    [0, 200],
    [0, 0],
    [1200, 50],
    [1200, 500]
])

def main():
    global result_csv_path, window_start_time, data_for_3_seconds, zone_coordinates
    window_start_time = time.time()
    data_for_3_seconds = []
    
    st.set_page_config(page_title="Project FlowCast", page_icon="📊", layout="wide")
    
    

    st.title("Project FlowCast")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Crowd Flow Analysis and Predictive Modeling for Infrastructure Management</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


  
  
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Save the uploaded video file to a local path
        MALL_VIDEO_PATH = "uploaded_video.mp4"
        with open(MALL_VIDEO_PATH, "wb") as f:
            f.write(uploaded_file.read())

        #Video instance for polygon zone setting
        cap = cv2.VideoCapture(MALL_VIDEO_PATH)
        video_info = sv.VideoInfo.from_video_path(MALL_VIDEO_PATH)      
        ret, frame = cap.read()

        if ret:
            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.sidebar.markdown("<span style='font-size: 35px; font-weight: bold;'>Settings :gear:</span>", unsafe_allow_html=True)
            st.sidebar.divider()
            st.sidebar.title("Video Info & Reference frame for polygon zone")
            
            st.sidebar.text(
                f"Resolution: {video_info.width}x{video_info.height}px\n"
                f"Total Frames: {video_info.total_frames}\n"
                f"Frame Rate (fps): {video_info.fps}"
            )
            st.sidebar.pyplot(fig)
            st.sidebar.divider()

            # Initiate polygon zone
            with st.sidebar.form("update_zone_form"):
                st.write("Update Polygon Zone Coordinates:")
                labels = ['A', 'B', 'C', 'D']
                for i in range(4):
                    st.write(f"Point {labels[i]}:")
                    for j in range(2):
                        key = f"coordinate_{i}_{j}"  # Unique key for each st.number_input
                        zone_coordinates[i, j] = st.number_input(
                            f"Coordinate {j + 1}", value=zone_coordinates[i, j], key=key
                        )
                submit_button = st.form_submit_button(label='Update Coordinates')

            # Check if the form is submitted
            if submit_button:
                st.success("Polygon Zone Coordinates Updated!")

            colors = sv.ColorPalette.default()
            video_info = sv.VideoInfo.from_video_path(MALL_VIDEO_PATH)
            zone = sv.PolygonZone(polygon=zone_coordinates, frame_resolution_wh=video_info.resolution_wh)

            zone_annotator = sv.PolygonZoneAnnotator(
                zone=zone,
                color=colors.by_idx(0),  # Use color from the palette
                thickness=6,
                text_thickness=8,
                text_scale=4
            )

            box_annotator = sv.BoxAnnotator(
                color=colors.by_idx(0),  # Use color from the palette
                thickness=4,
                text_thickness=4,
                text_scale=2
            )

        else:
            st.sidebar.error("Error reading the video frame.")

        # Assuming 25 frames per second
        frames_per_second = video_info.fps
        frames_per_interval = 3 * frames_per_second
        
        # CSV file setup
        csv_file_path = "/content/detections_count.csv"

        def process_frame(frame: np.ndarray, i) -> np.ndarray:
          global data_for_3_seconds, window_start_time, data_for_3_seconds 

          #detect
          results = model(frame, size=video_info.width)
          detections = sv.Detections.from_yolov5(results)
          detections = detections[(detections.class_id == 0) & (detections.confidence > 0.2)]

          # Count detections inside the polygon zone
          mask = zone.trigger(detections=detections)
          detections_inside_zone = detections[mask]
          num_detections_inside_zone = len(detections_inside_zone)

          # Print the number of detections inside the polygon zone for every consecutive 3 seconds
          if i % frames_per_interval == 0:
            seconds = i / frames_per_second
            # Use time module to get the current timestamp only once
            current_timestamp = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")

            # Check if 3 seconds have elapsed to update the timestamp
            elapsed_time = time.time() - window_start_time
            if elapsed_time >= 3:
              timestr = datetime.datetime.fromtimestamp(window_start_time, pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d %H:%M:%S")
              # Save data for the current 3-second window
              data_for_3_seconds.append({
                'Timestamp': timestr,
                'Crowd_Counts': num_detections_inside_zone,
                #'Timestamp': current_timestamp  # Store the timestamp for the 3-second window
              })
              # Print and store the information
              print(f"Number of detections inside the zone (Time {seconds:.2f} seconds): {num_detections_inside_zone}")
              
              #increment time by 3sec
              window_start_time +=3
              
              

              # Append data to CSV file
              with open(csv_file_path, 'a', newline='') as csvfile:
                fieldnames = ['Timestamp', 'Crowd_Counts']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # Write header if the file is newly created
                if csvfile.tell() == 0:
                  writer.writeheader()

                writer.writerow({ 'Timestamp': timestr, 'Crowd_Counts': num_detections_inside_zone})  

          frame = box_annotator.annotate(scene=frame, detections=detections_inside_zone, skip_label=True)
          frame = zone_annotator.annotate(scene=frame)    


          return frame



        if st.button("Process Video"):
          #Starting of processing
          result_video_path_colab = "/content/market-square-result.mp4"
          sv.process_video(source_path=MALL_VIDEO_PATH, target_path=result_video_path_colab, callback=process_frame)

          #detections dataframe
          df = pd.read_csv(csv_file_path)
          
          with st.container():
            container = st.container(border=True)
            container.write("Crowd Count Storage DataFrame:")
            container.write(df)

            if not df.empty:

               #Load the saved Random forest model from pickle file
               with open('rd.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file)

            
                # Assume 'Crowd_Counts' is the column containing crowd count data
                crowd_counts = df['Crowd_Counts'].values.reshape(-1, 1)

                # Make predictions using the loaded model
                predicted_crowd_types = loaded_model.predict(crowd_counts)
                

                # Create a new DataFrame with original data, predicted crowd types, and timestamp
                r_df = pd.DataFrame({
                    'Predicted_Crowd_Type': predicted_crowd_types   
                })

                #Define the crowd type mapping
                crowd_type_mapping = {0: 'High Crowd', 1: 'Low Crowd', 2:'Moderate Crowd'} 
                # Replace numerical values using replace method
                result_df = pd.DataFrame({
                    'Timestamp': pd.to_datetime(df['Timestamp']),  # Convert Timestamp to datetime format
                    'Crowd_Counts': df['Crowd_Counts'],
                    'Predicted_Crowd_Type': r_df['Predicted_Crowd_Type'].replace(crowd_type_mapping)                  
                })

                #save the dataframe
                result_df.to_csv('/content/result_df.csv', index=False)


                #Crowd Management suggestions
                high_crowd = len(result_df[result_df['Predicted_Crowd_Type']== 'High Crowd'])
                low_crowd = len(result_df[result_df['Predicted_Crowd_Type']== 'Low Crowd'])
                moderate_crowd = len(result_df[result_df['Predicted_Crowd_Type']== 'Moderate Crowd'])
                


                with st.container():
                  container = st.container(border=True)
                  col1, col2 = container.columns([3,1])
                  col1.write("Predicted Crowd Type DataFrame:")
                  col1.write(result_df)
                  col2.write(" Machine Learning Model For Processing:")
                  col2.write("Random Forest")


                  max_count = max(low_crowd, moderate_crowd, high_crowd)
                  if max_count == low_crowd:
                    container.warning('Low Crowd observed no additional services required')
                  elif max_count == moderate_crowd:
                    container.warning('Moderate Crowd detected , may consider ensuring proper services')
                  elif max_count == high_crowd:
                    container.warning('High Crowd detected, additional management and services required!')

                  #Peak Hours
                  if 'Predicted_Crowd_Type' in result_df.columns:
                    moderate_mask = result_df['Predicted_Crowd_Type'] == 'Moderate Crowd'
                    high_mask = result_df['Predicted_Crowd_Type'] == 'High Crowd'

                     # Check if there are moderate crowd data
                    if moderate_mask.any():
                      moderate_peak_timestamps = result_df[moderate_mask][result_df[moderate_mask]['Crowd_Counts'] == result_df[moderate_mask]['Crowd_Counts'].max()]['Timestamp'].tolist()
                      moderate_peak_df = pd.DataFrame({'Peak Moderate Crowd were found at': moderate_peak_timestamps})
                    else:
                      moderate_peak_df = None


                    # Check if there are high crowd data
                    if high_mask.any():
                      high_peak_timestamps = result_df[high_mask][result_df[high_mask]['Crowd_Counts'] == result_df[high_mask]['Crowd_Counts'].max()]['Timestamp'].tolist()
                      high_peak_df = pd.DataFrame({'Peak High Crowd were found at': high_peak_timestamps})
                    else:
                      high_peak_df = None
 

                    if moderate_peak_df is not None:
                       container.write(moderate_peak_df)
                    else:
                      container.info("No peak Moderate Crowds found")

                    if high_peak_df is not None:
                      container.write(high_peak_df)
                    else:
                      container.info("No peak High Crowds found")     

                  else:    
                    container.warning('Error Loading Data')


                # Monthly Trend Line Chart
                with st.container():
                  container = st.container(border=True)

                if 'Timestamp' in result_df.columns:
                  df2 = result_df.copy()
                  sns.set(style="dark")
                  plt.figure(figsize=(12, 6))
                  sns.lineplot(x='Timestamp', y='Crowd_Counts', hue='Predicted_Crowd_Type', marker='o', linestyle='-', data=df2)
                  plt.title('Crowd Count Over Time with Predicted Crowd Types')
                  plt.xlabel('Timestamp')
                  plt.ylabel('Crowd Counts')
                  plt.xticks(rotation=45)
                  plt.grid(True)
                  container.pyplot(plt)

            except pd.errors.EmptyDataError:
                st.warning("The CSV file is empty. Please upload a valid CSV file with data.")
            
            else:
              html_temp = """
              <div style="background-color:teal ;padding:10px">
              <h2 style="color:white; text-align:center;">Project Tranining History</h2>
              </div>
              """
              st.markdown(html_temp, unsafe_allow_html=True)
              st.info('The Project was Trained using Random Forest Model with Accuracy score 0.96 and F1 score of 0.94 ')
              df2 = pd.read_csv(github_csv, parse_dates=["Timestamp"])
              if not df2.empty:
                with st.container():
                  container =  st.container(border=True)
                  container.write("Project Training Data:")
                  container.write(df2)
                  container.divider()
                  
                  container.title("Crowd Count Visualization & Monthly Average Trend Line")

                  tab1, tab2 = container.tabs(["Crowd Count Visualization","Monthly Average Trend Line"])
                  #Plotting
                with tab1: 
                  df2 = pd.read_csv(github_csv, parse_dates=["Timestamp"])

                  
                  chart = alt.Chart(df2).mark_line().encode(
                    x = "Timestamp:T",
                    y = "Noisy_Crowd_Count:Q",
                    color = "Crowd_Type:N",
                    tooltip = ["Timestamp:T", "Noisy_Crowd_Count:Q", "Crowd_Type:N"],
                  ).properties(width=1800, height=400)
                  tab1.altair_chart(chart, use_container_width=True)
                  
                with tab2:
                  sns.set(style="darkgrid")
                  monthly_avg = df2.groupby('Month_Name')['Noisy_Crowd_Count'].mean()
                  fig, ax = plt.subplots(figsize=(12,6))
                  ax.plot(monthly_avg.index, monthly_avg, marker='o', linestyle='-')
                  ax.set_title('Average Trend line')
                  ax.grid(True)
                  tab2.pyplot(fig)
                  ##
              else:
                  container.warning(f"Failed to retrieve data.")



if __name__ == '__main__':
    main()

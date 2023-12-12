import streamlit as st
import subprocess
# Install required packages using subprocess
subprocess.run(["pip", "install", "pandas", "ultralytics", "supervision==0.2.0"])
# Install detectron2 from GitHub
subprocess.run(["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import supervision as sv
import pandas as pd
import datetime
import pytz
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler





# Declare max_persons and result_csv_path as global variables
max_persons = 0
result_csv_path = None

def main():
    global result_csv_path  # Declare result_csv_path as global

    st.title("Project FlowCast")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Crowd Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities = ['Decision tree', 'Random forest', 'XGBoost', 'Tuned XGBoost']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)

    # Load YOLO model
    model = YOLO('yolov8s.pt')

    # Initialize variables for video length, start timestamp, and max persons
    video_length = 0  # Initialize to 0

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        # Save the uploaded video file to a local path
        MALL_VIDEO_PATH = "uploaded_video.mp4"
        with open(MALL_VIDEO_PATH, "wb") as f:
            f.write(uploaded_file.read())

        st.text("Video file saved to: {}".format(MALL_VIDEO_PATH))

        # Initiate polygon zone
        polygon = np.array([
            [0, 200],
            [0, 0],
            [1200, 50],
            [1200, 500]
        ])
        video_info = sv.VideoInfo.from_video_path(MALL_VIDEO_PATH)
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

        # Initiate annotators
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

        # Initialize a global list to store information about detected persons
        person_data = []

        def process_frame(frame: np.ndarray, frame_index: int) -> np.ndarray:
            global max_persons, result_csv_path  # Declare max_persons and result_csv_path as global

            # detect
            results = model(frame, imgsz=1280)[0]
            detections = sv.Detections.from_yolov8(results)
            detections = detections[detections.class_id == 0]  # Assuming class_id 0 corresponds to persons
            zone.trigger(detections=detections)

            # Update max persons count
            frame_person_count = len(detections)
            max_persons = max(max_persons, frame_person_count)

            # Update video length based on frame index
            video_length = frame_index / video_info.fps  # Calculate video length in seconds

            # Append information about detected persons to the list
            for _ in detections:
                label = f"Person {len(person_data) + 1}"
                person_data.append({'Timestamp': frame_index, 'Label': label})

            # annotate
            labels = [f"Person {i + 1}" for i in range(len(detections))]
            annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)

            return annotated_frame

            result_csv_path = None

        if st.button("Process Video"):
            # Process the entire video and save the result
            result_video_path_colab = "/content/result2.mp4"
            sv.process_video(source_path=MALL_VIDEO_PATH, target_path=result_video_path_colab, callback=process_frame)

            # Display the processed video
            # st.video(result_video_path_colab)

            # Print the total number of persons detected, video length, and max persons
            print("Total Number of Persons Detected(detection sum count from all frames):", len(person_data))
            print("Max Persons Detected in a Frame:", max_persons)
            st.write("Max Persons Detected :", max_persons)

            # Get the current timestamp in the local time zone
            local_timezone = pytz.timezone('Asia/Kolkata')  # Adjust the time zone according to your location
            end_timestamp_local = datetime.datetime.now(tz=local_timezone)

            # Format timestamp as "01-01-2023 12:00:00 PM"
            formatted_timestamp = end_timestamp_local.strftime("%m-%d-%Y %I:%M:%S %p")

            # Create a DataFrame with max_persons count and timestamp
            result_df = pd.DataFrame({
                'Timestamp': [formatted_timestamp],
                'Crowd_Count': [max_persons]
            })

            # Save the DataFrame to a CSV file
            result_csv_path = "/content/crowd_count_result.csv"
            result_df.to_csv(result_csv_path, index=False)

            # Check the selected option for further actions
            if option == 'Decision tree':
                # Load the saved model from the pickle file
                with open('dect.pkl', 'rb') as model_file:
                    loaded_model = pickle.load(model_file)

                # User input for the crowd number
                new_noisy_crowd_count = max_persons

                # Reshape the input for prediction (assuming single feature)
                new_noisy_crowd_count_reshaped = [[new_noisy_crowd_count]]

                # Make prediction using the loaded model
                predicted_crowd_type = loaded_model.predict(new_noisy_crowd_count_reshaped)

                # Display the predicted crowd type
                st.write("Observed Crowd Type:", predicted_crowd_type[0])
    
            elif option == 'Random forest':
                #Loading pkl file
                with open('rd.pkl', 'rb') as model_file:
                  loaded_model=pickle.load(model_file) 

                #user input crowd number
                new_crowd_count = [[max_persons]]  

                # Make prediction using the loaded model
                predicted_crowd_type_rd = loaded_model.predict(new_crowd_count)
                st.write("Observed Crowd Type:", predicted_crowd_type_rd[0])

            elif option == 'XGBoost':
               #Load the ML model
              with open('xgb.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file) 

                #Input
                new_noisy_crowd_count_xg = max_persons
                #Reshaping the input
                new_noisy_crowd_count_reshaped_xg = [[new_noisy_crowd_count_xg]]
                predicted_numerical_label = loaded_model.predict(new_noisy_crowd_count_reshaped_xg)[0]
                # Convert the numerical prediction to string label using the mapping
                # Map numerical labels to string labels
                numerical_to_string_mapping = {1: 'Low Crowd', 2: 'Moderate Crowd', 0: 'High Crowd'}
                # Convert the numerical prediction to string label using the mapping
                predicted_crowd_type_xg = numerical_to_string_mapping[predicted_numerical_label]

              st.write("Observed Crowd Type:", predicted_crowd_type_xg)

            elif option == 'Tuned XGBoost':
               #Load the ML model
              with open('xgboost_tuned.pkl', 'rb') as model_file:
                loaded_model = pickle.load(model_file)

                #input
                new_noisy_crowd_count_xgbt = max_persons

                # Reshape the input for prediction (assuming single feature)
                new_noisy_crowd_count_reshaped_xgbt = [[new_noisy_crowd_count_xgbt]]
                # Make prediction using the loaded model
                predicted_numerical_label = loaded_model.predict(new_noisy_crowd_count_reshaped_xgbt)[0]

                # Map numerical labels to string labels
                numerical_to_string_mapping = {1: 'Low Crowd', 2: 'Moderate Crowd', 0: 'High Crowd'}
                # Convert the numerical prediction to string label using the mapping
                predicted_crowd_type_xgbt = numerical_to_string_mapping[predicted_numerical_label]

                # Display the predicted crowd type
                st.write("Observed Crowd Type:", predicted_crowd_type_xgbt)

        

if __name__ == '__main__':
    main()
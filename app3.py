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



# Load the models
dec = pickle.load(open('dec.pkl', 'rb'))
rf = pickle.load(open('rf.pkl', 'rb'))
xg = pickle.load(open('xg.pkl', 'rb'))
best_xgb = pickle.load(open('best_xgb.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Declare max_persons and result_csv_path as global variables
max_persons = 0
result_csv_path = None

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
    global result_csv_path  # Declare result_csv_path as global

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

    # Install required packages using subprocess
    subprocess.run(["pip", "install", "pandas", "ultralytics", "supervision==0.2.0"])
    # Install detectron2 from GitHub
    subprocess.run(["pip", "install", "git+https://github.com/facebookresearch/detectron2.git"])

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

    # Assuming the column containing crowd counts is named 'Crowd_Count'
    if result_csv_path is not None:
        data = pd.read_csv(result_csv_path)

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

            overall_avg = np.mean(crowd_counts)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=timestamps, y=crowd_counts, hue=results, palette="viridis", ax=ax, label='Crowd Count')
            sns.lineplot(x=daily_avg.index, y=daily_avg['Crowd_Count'], color='blue', label='Daily Avg', ax=ax)
            sns.lineplot(x=weekly_avg.index, y=weekly_avg['Crowd_Count'], color='green', label='Weekly Avg', ax=ax)
            sns.lineplot(x=monthly_avg.index, y=monthly_avg['Crowd_Count'], color='orange', label='Monthly Avg', ax=ax)
            plt.axhline(overall_avg, color='red', linestyle='--', label='Overall Avg')
            plt.title('Time vs Crowd Count with Average Estimation')
            plt.xlabel('Timestamp')
            plt.ylabel('Crowd Count')
            plt.legend()
            st.pyplot(fig)


            results_df = pd.DataFrame({'Timestamp': timestamps, 'Crowd_Count': crowd_counts, 'Classification': results})
            st.write(results_df)

        else:
            st.error("Please make sure your CSV file has columns named 'Timestamp' and 'Crowd_Count.'")

if __name__ == '__main__':
    main()

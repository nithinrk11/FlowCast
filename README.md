# FlowCast
## Crowd flow Analysis and Predictive Modelling for Infrastructure Management

### Effective crowd control is crucial for improving total public space use and resource allocation in today's urban landscapes. The "Project FlowCast" is an all-inclusive solution that gives insights into crowd dynamics using computer vision and machine learning. The project aims to help infrastructure management with well-informed judgments by using cutting-edge methods to identify, categorize, and evaluate crowds inside predefined zones.


## Overview
**Project FlowCast** is designed for real-time crowd flow analysis and predictive modeling to aid infrastructure management. By leveraging advanced machine learning models and real-time video processing, it provides valuable insights into crowd density and types within specified zones, helping in efficient crowd management and planning.

## Features
- **Real-time Video Processing:** Detects and counts the number of people within a specified zone in a video.
- **Dynamic Zone Configuration:** Allows users to define and update a polygonal zone for monitoring.
- **Predictive Modeling:** Uses a pre-trained Random Forest model to predict crowd types based on detected counts.
- **Data Visualization:** Visualizes the detected counts and predicted crowd types over time.
- **CSV Data Handling:** Processes, stores, and displays data from CSV files.

## Components

### 1. Installation of Required Packages
The necessary Python packages are installed, including Pandas for data manipulation, Ultralytics for YOLO model, Supervision for video annotation, and additional packages for data visualization and machine learning.

### 2. Loading the YOLO Model
A YOLO model (YOLOv5) is loaded from the Ultralytics repository. This model is used for real-time object detection within the video frames.

### 3. Setting Detection Zone Coordinates
A polygonal zone is defined within which the crowd is monitored. The initial coordinates of the zone are specified, but these can be dynamically updated by the user through the application's sidebar interface.

### 4. Streamlit Application Setup
Streamlit is used to create an interactive web application. The application is set up with a title and layout configuration, and an introductory banner is displayed at the top.

### 5. Video Upload and Display
Users can upload a video file in mp4 or avi format. The uploaded video is saved locally and displayed in the application interface.

### 6. Initial Frame and Zone Display
An initial frame from the uploaded video is captured and displayed in the sidebar. This frame serves as a reference for setting and updating the polygonal zone coordinates.

### 7. Dynamic Zone Configuration
A form in the sidebar allows users to update the coordinates of the polygonal detection zone. Users can input the x and y coordinates for each point of the polygon, and upon submission, the zone is updated and a success message is displayed.

### 8. Frame Processing and Object Detection
The uploaded video is processed frame by frame:
- The YOLO model detects objects in each frame.
- Only persons (class ID 0) with confidence above 20% are considered.
- The number of detected persons within the polygonal zone is counted.
- Every three seconds, the count is recorded along with a timestamp and stored in a CSV file.

### 9. Video Processing and Data Storage
Once the video processing is initiated, the frame processing function is applied to each frame of the video. The processed frames are annotated with detection results and saved in a new video file. Detected counts and timestamps are also appended to a CSV file for further analysis.

### 10. Displaying Detection Results
After processing, the detected counts are displayed in a data table. The data is also used to make predictions about the type of crowd (high, low, moderate) using a pre-trained Random Forest model.

### 11. Crowd Type Prediction
The pre-trained Random Forest model predicts the crowd type based on the detected counts. The predictions, along with timestamps and counts, are stored in a new DataFrame, which is displayed in the application.

### 12. Crowd Management Suggestions
Based on the predicted crowd types, suggestions are provided for crowd management:
- If the majority of the data indicates a high crowd, a warning is displayed suggesting the need for additional management and services.
- For moderate crowd levels, it suggests ensuring proper services.
- For low crowd levels, it indicates no additional services are required.

### 13. Peak Hours Identification
The timestamps corresponding to peak moderate and high crowd counts are identified and displayed, helping in understanding peak crowd hours.

### 14. Data Visualization
A line chart is plotted to visualize crowd counts over time, colored by the predicted crowd type. This helps in observing trends and fluctuations in crowd density.

### 15. Training Data Display
The application also includes a section displaying historical training data used to train the model. This data is visualized to show monthly trends and average crowd counts, providing insights into historical crowd patterns.

## Conclusion
Project FlowCast offers a robust solution for real-time crowd monitoring and predictive analysis. By combining video processing with machine learning and interactive data visualization, it helps in making informed decisions for efficient crowd management and infrastructure planning.

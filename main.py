import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import numpy as np
import logging
from datetime import datetime
import sqlite3

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Initialize YOLO model and custom tracker
model = YOLO('yolov8s.pt')
tracker = Tracker()

# Set the source of the video stream (IP camera URL or video file path)
source = 'video/1.mp4'  # Update to your video source

# Initialize video capture from the source
cap = cv2.VideoCapture(source)

# Check if the video capture was initialized successfully
if not cap.isOpened():
    print(f"Error: Unable to open video source {source}")
    exit()

# Load class list from file
with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")

# Define the areas for tracking
area1 = [(494, 289), (505, 499), (578, 496), (530, 292)]
area2 = [(548, 290), (600, 496), (637, 493), (574, 288)]

# Initialize dictionaries and counters
person_log = {}
going_out = {}
going_in = {}
counter_out = []
counter_in = []

# Set up SQLite database connection
conn = sqlite3.connect('person_movement.db')
cursor = conn.cursor()

# Create a table to store movement logs
cursor.execute('''
    CREATE TABLE IF NOT EXISTS movement_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER,
        timestamp TEXT,
        state TEXT
    )
''')
conn.commit()

# Function to log person movement in the SQLite database
def log_movement(person_id, state):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO movement_log (person_id, timestamp, state) VALUES (?, ?, ?)", (person_id, current_time, state))
    conn.commit()

# Main loop for processing video frames
while True:    
    ret, frame = cap.read()
    if not ret:
        print("Unable to read video frame or stream ended.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO model prediction on the current frame
    results = model.predict(frame)
    predictions = results[0].boxes.data
    prediction_df = pd.DataFrame(predictions).astype("float")

    # Extract coordinates of detected persons
    coordinates = []
    for _, row in prediction_df.iterrows():
        x1, y1, x2, y2, _, class_id = row.astype(int)
        class_name = class_list[class_id]
        
        if 'person' in class_name:
            coordinates.append([x1, y1, x2, y2])

    # Update tracker with detected coordinates
    tracked_objects = tracker.update(coordinates)

    # Process each tracked object
    for obj in tracked_objects:
        x3, y3, x4, y4, obj_id = obj

        # Check if the object is moving out (from area2 to area1)
        if cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
            going_out[obj_id] = (x4, y4)
        
        if obj_id in going_out:
            if cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
                # Log movement if it's the first time the object is outside
                if obj_id not in counter_out:
                    counter_out.append(obj_id)
                    person_log[obj_id] = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "state": "outside"}
                    log_movement(obj_id, "outside")

                # Draw circle and rectangle around object
                cv2.circle(frame, (x4, y4), 8, (255, 0, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cv2.putText(frame, f'{obj_id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Check if the object is moving in (from area1 to area2)
        if cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
            going_in[obj_id] = (x4, y4)

        if obj_id in going_in:
            if cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
                # Log movement if it's the first time the object is inside
                if obj_id not in counter_in:
                    counter_in.append(obj_id)
                    person_log[obj_id] = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "state": "inside"}
                    log_movement(obj_id, "inside")

                # Draw circle and rectangle around object
                cv2.circle(frame, (x4, y4), 8, (255, 0, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cv2.putText(frame, f'{obj_id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display counts of objects inside and outside the areas
    outside_count = len(counter_out)
    inside_count = len(counter_in)
    cv2.putText(frame, f'Outside: {outside_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, f'Inside: {inside_count}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Draw polygons for the areas
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("RGB", frame)

    # Exit on pressing 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Close the SQLite connection
conn.close()

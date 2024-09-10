# Object Tracking and Movement Logging System

## Overview

This project implements an object tracking and movement logging system using the YOLO (You Only Look Once) object detection model. The system tracks individuals in a video feed, monitors their movement across predefined areas, and logs this movement into an SQLite database.

## Features

- **Object Detection**: Uses YOLO model to detect objects in video frames.
- **Object Tracking**: Tracks detected objects across frames with unique IDs.
- **Movement Logging**: Logs the movement of tracked individuals into an SQLite database.
- **Area Monitoring**: Monitors and logs if individuals move between predefined areas.

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- Pandas (`pandas`)
- NumPy (`numpy`)
- SQLite (`sqlite3`)
- Ultralytics YOLO (`ultralytics`)

## Setup

1. **Clone the Repository**

   ```sh
   git clone https://github.com/realtourani/Object-Tracking-and-Movement-Logging-System.git
   cd Object-Tracking-and-Movement-Logging-System-main

## Install Dependencies
Install the required Python libraries using pip:
```sh
  pip install opencv-python-headless pandas numpy sqlite3 ultralytics
  ```

## Configuration
1. Video Source

   Update the `source` variable in `main.py` with the path to your video file or the URL of your IP camera.

2. Define Tracking Areas

   Modify the `area1` and `area2` variables to define the areas you want to monitor. These areas are defined by a list of coordinates forming polygons.

## Usage
1. Run the Script
  Execute the main script to start the object tracking and movement logging process:
  ```sh
    python main.py
  ```
2. View Output
The processed video frames will be displayed in a window. The system will log the movement of individuals between the defined areas. Press `Esc` to exit the video feed window.

3. Access Logs
Movement logs are stored in an SQLite database file named `person_movement.db`. The table `movement_log` contains records of person IDs, timestamps, and their movement states.









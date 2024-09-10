import math

class Tracker:
    def __init__(self):
        # Dictionary to store center positions of tracked objects with their IDs
        self.center_points = {}
        # Counter for assigning unique IDs to newly detected objects
        self.id_count = 0

    def update(self, object_rects):
        # List to store bounding boxes and corresponding IDs
        tracked_objects = []

        # Iterate through all detected object rectangles
        for rect in object_rects:
            x, y, w, h = rect
            cx = (x + x + w) // 2  # Calculate center x-coordinate
            cy = (y + y + h) // 2  # Calculate center y-coordinate

            # Flag to check if the object is already detected
            same_object_detected = False

            # Check if the object matches an existing tracked object
            for object_id, center in self.center_points.items():
                dist = math.hypot(cx - center[0], cy - center[1])  # Calculate distance between centers

                if dist < 35:  # If the distance is small enough, consider it the same object
                    self.center_points[object_id] = (cx, cy)
                    tracked_objects.append([x, y, w, h, object_id])
                    same_object_detected = True
                    break

            # If no match is found, assign a new ID to the object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                tracked_objects.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Remove unused object IDs from the dictionary
        updated_center_points = {obj_id: self.center_points[obj_id] for _, _, _, _, obj_id in tracked_objects}

        # Update the center points dictionary
        self.center_points = updated_center_points
        return tracked_objects

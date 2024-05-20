import cv2

def check_available_cameras():
    # Maximum number of cameras to check
    max_cameras = 10

    # Iterate through camera indices
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        
        # Check if the camera is opened successfully
        if cap.isOpened():
            print(f"Camera {i} is available")
            cap.release()  # Release the VideoCapture object
        else:
            print(f"Camera {i} is not available")

if __name__ == "__main__":
    check_available_cameras()

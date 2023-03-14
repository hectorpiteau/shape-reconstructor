import cv2


# List available cameras
num_cameras = 0
while True:
    cap = cv2.VideoCapture(num_cameras)
    if not cap.read()[0]:
        break
    else:
        print(f"Camera {num_cameras} is available")
    cap.release()
    num_cameras += 1

# Open the camera
cap = cv2.VideoCapture(2)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

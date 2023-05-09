import cv2
import time


def openCameraImageClick():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Wait for 5 seconds to ensure the camera is ready
    time.sleep(5)

    # Capture an image
    ret, frame = cap.read()

    # Release the camera
    cap.release()

    # Save the image
    # global img_saved
    cv2.imwrite("captured_image.jpg", frame)

    # Display the captured image
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# openCameraImageClick()

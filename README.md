import cv2

# Load the hat image
hat_img = cv2.imread('hat.png', -1)

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to overlay hat on detected faces
def overlay_hat(frame, hat, x, y, w, h):
    # Resize the hat to fit the face width
    resized_hat = cv2.resize(hat, (w, h))

    # Get the region of interest on the frame for the hat
    roi = frame[y:y+h, x:x+w]

    # Create a mask and inverse mask of the hat
    hat_gray = cv2.cvtColor(resized_hat, cv2.COLOR_BGR2GRAY)
    _, hat_mask = cv2.threshold(hat_gray, 25, 255, cv2.THRESH_BINARY_INV)

    # Mask the region of interest to create the effect of the hat
    hat_area = cv2.bitwise_and(resized_hat, resized_hat, mask=hat_mask)
    masked_area = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(hat_mask))

    # Combine the hat area and the original frame
    combined_roi = cv2.add(masked_area, hat_area)

    # Update the original frame with the combined region of interest
    frame[y:y+h, x:x+w] = combined_roi

    return frame

# Main function to capture video from webcam and apply filter
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Apply filter to each detected face
        for (x, y, w, h) in faces:
            frame = overlay_hat(frame, hat_img, x, y, w, h)

        # Display the frame
        cv2.imshow('Snapchat Filter', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

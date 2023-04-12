import cv2

# Load image
image = cv2.imread('Donald.jpg')

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load Haar Cascade classifier for eye detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# Loop over face and detect the eyes
for (x, y, w, h) in faces:
    # Extract the region of interest (ROI) corresponding to the face
    face_roi = gray[y:y+h, x:x+w]

    # Detect eyes in the face ROI
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Loop over each eye and draw a black rectangle over it
    for (ex, ey, ew, eh) in eyes:
        # Compute the coordinates of the eye in the original image
        eye_x = x + ex
        eye_y = y + ey

        # Draw a black rectangle over the eye
        cv2.rectangle(image, (eye_x, eye_y), (eye_x+ew, eye_y+eh), (0, 0, 0), -1)

#save the image
#cv2.imwrite('output_image.jpg', image)


# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)

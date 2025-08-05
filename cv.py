import cv2
import numpy as np
import tensorflow as tf

# Load your saved model
model = tf.keras.models.load_model("digit_model.h5")

# Create a white canvas
canvas = np.ones((400, 400), dtype='uint8') * 255
drawing = False

# Mouse callback function
def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 8, (0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Draw a Digit')
cv2.setMouseCallback('Draw a Digit', draw)

while True:
    cv2.imshow('Draw a Digit', canvas)
    key = cv2.waitKey(1)

    if key == ord('c'):  # Clear canvas
        canvas.fill(255)

    elif key == ord('p'):  # Predict
        # Resize to 28x28 and invert (black digit on white)
        roi = cv2.resize(canvas, (28, 28))
        roi = 255 - roi
        roi = roi / 255.0
        roi = roi.reshape(1, 28, 28, 1)

        prediction = model.predict(roi)
        predicted_digit = np.argmax(prediction)

        print(f"Predicted Digit: {predicted_digit}")
        cv2.putText(canvas, f'Predicted: {predicted_digit}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0), 2)

    elif key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()

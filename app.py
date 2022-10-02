from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

cap = cv2.VideoCapture(0)


@app.route('/init', methods=['POST', 'GET'])
def init():
    global cap
    if request.method == 'POST':
        if request.form.get('start') == 'Start':
            cap = cv2.VideoCapture(0)
            return redirect(url_for('index'))
        if request.form.get('stop') == 'Stop':
            cap.release()
            cv2.destroyAllWindows()
            return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))


prediction_class_converter = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "K",
    10: "L",
    11: "M",
    12: "N",
    13: "O",
    14: "P",
    15: "Q",
    16: "R",
    17: "S",
    18: "T",
    19: "U",
    20: "V",
    21: "W",
    22: "X",
    23: "Y"
}

prec_image = str()


def predict_class_image(frame_gray):
    global prec_image
    gray_image = cv2.merge((frame_gray, frame_gray, frame_gray))

    gray_image_resized = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    gray_image_test_normalized = gray_image_resized / 255

    gray_image_test_normalized = (np.expand_dims(gray_image_test_normalized, 0))

    sign_model = load_model('sign_language_interpreter_model.h5')
    prediction_class = np.argmax(sign_model.predict(gray_image_test_normalized))

    class_predict = prediction_class_converter[prediction_class]

    # no need to print the same class many times
    if prec_image == class_predict:
        return

    prec_image = class_predict

    return class_predict


def frames_generator():
    if not cap.isOpened():
        print('Error opening video')
        exit(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            class_predicted = predict_class_image(frame_gray)
            cv2.putText(frame, class_predicted, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA);
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video', methods=['POST', 'GET'])
def video():
    return Response(frames_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

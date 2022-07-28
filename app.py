from flask import Flask, render_template, Response, request, redirect, url_for
import cv2

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


def frames_generator():
    if not cap.isOpened():
        print('Error opening video')
        exit(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        else:
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

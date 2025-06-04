# app.py - Web version of Hand Gesture System
from flask import Flask, render_template, request, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import threading
import time
from io import BytesIO
import os

app = Flask(__name__)

class WebGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_image(self, image_data):
        """Process base64 encoded image and detect gestures"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return None, []

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            gestures = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    # Simple gesture detection
                    landmarks = hand_landmarks.landmark

                    # Point gesture (index finger up)
                    if landmarks[8].y < landmarks[6].y < landmarks[5].y:
                        gestures.append({
                            'name': 'point',
                            'x': landmarks[8].x,
                            'y': landmarks[8].y
                        })

                    # Click gesture (thumb and index close)
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    distance = ((thumb_tip.x - index_tip.x) ** 2 +
                                (thumb_tip.y - index_tip.y) ** 2) ** 0.5

                    if distance < 0.05:
                        gestures.append({'name': 'click'})

                    # Peace sign (index and middle up)
                    if (landmarks[8].y < landmarks[6].y and
                            landmarks[12].y < landmarks[10].y):
                        gestures.append({'name': 'peace'})

            # Encode processed frame back to base64
            _, buffer = cv2.imencode('.jpg', frame)
            processed_image = base64.b64encode(buffer).decode('utf-8')

            return processed_image, gestures

        except Exception as e:
            print(f"Error processing image: {e}")
            return None, []

# Global recognizer instance
recognizer = WebGestureRecognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_gesture', methods=['POST'])
def process_gesture():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        processed_image, gestures = recognizer.process_image(image_data)

        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 500

        return jsonify({
            'success': True,
            'processed_image': f"data:image/jpeg;base64,{processed_image}",
            'gestures': gestures
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
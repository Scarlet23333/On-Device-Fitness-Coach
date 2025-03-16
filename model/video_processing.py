from utils.math_utils import calculate_angle
from model_compile import pose_detector_target_model, pose_landmark_detector_target_model

import eventlet

eventlet.monkey_patch()

from flask import Flask, Response
from flask_socketio import SocketIO
import cv2
import torch

app = Flask(__name__)
socketio = SocketIO(app)

# Use the compiled models for real-time on-device execution
pose_detector = pose_detector_target_model  # Use compiled model
pose_landmark_detector = pose_landmark_detector_target_model


def generate_frames():
    cap = cv2.VideoCapture(0)
    squat_count = 0
    squat_in_progress = False
    added_count = False
    knee_angle_threshold = 90
    knee_angle_threshold_low = 60
    hip_angle_threshold = 130
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.tensor(rgb_frame, dtype=torch.float32).unsqueeze(0)

        # Run pose detection using compiled on-device model
        results = pose_detector(frame_tensor)
        if "pose_landmarks" in results:
            landmarks = results["pose_landmarks"]
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[23].x, landmarks[23].y]  # LEFT_HIP
            knee = [landmarks[25].x, landmarks[25].y]  # LEFT_KNEE
            ankle = [landmarks[27].x, landmarks[27].y]  # LEFT_ANKLE
            shoulder = [landmarks[11].x, landmarks[11].y]   # LEFT_SHOULDER
            knee_angle = calculate_angle(hip, knee, ankle)
            hip_angle = calculate_angle(knee, hip, shoulder)
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, model.POSE_CONNECTIONS)
            if results.pose_landmarks:
                for landmark in results.pose_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'Knee Angle: {int(knee_angle)} degrees', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Hip Angle: {int(hip_angle)} degrees', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Squat Count: {squat_count}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if hip_angle < hip_angle_threshold and not squat_in_progress:
                squat_in_progress = True
            elif hip_angle >= hip_angle_threshold and squat_in_progress:
                squat_in_progress = False
                if not added_count:
                    socketio.emit('posture_alert', {'message': 'Your knees were not bending enough!'})
                added_count = False

            if squat_in_progress:
                if knee_angle < knee_angle_threshold and not added_count:
                    squat_count += 1
                    added_count = True
                if knee_angle < knee_angle_threshold_low:
                    socketio.emit('posture_alert', {'message': 'Your hip is too low!'})

        # Resize the frame
        height, width, _ = frame.shape
        new_width = 560 # 560
        new_height = int(height * new_width / width)
        frame = cv2.resize(frame, (new_width, new_height))  # Resize to 1280x720 resolution (HD)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, debug=True)

"""
react use e.g.

import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';

const socket = io('http://localhost:5000');  // Change this URL if you deploy to a server

function App() {
  const [alert, setAlert] = useState('');
  const [showVideo, setShowVideo] = useState(false);

  useEffect(() => {
    // Listen for posture alerts from the backend
    socket.on('posture_alert', (data) => {
      setAlert(data.message);

      // Hide the alert after 5 seconds
      setTimeout(() => {
        setAlert('');
      }, 5000);  // Alert will disappear after 5 seconds
    });

    return () => {
      socket.off('posture_alert');
    };
  }, []);

  const toggleVideo = () => {
    setShowVideo(!showVideo); // Toggle the video feed visibility
  };

  return (
    <div className="App">
      <h1>Fitness Coach</h1>

      {/* Show the alert if it exists */}
      {alert && (
        <div style={{ 
          backgroundColor: '#f8d7da', 
          color: '#721c24', 
          padding: '10px', 
          marginBottom: '15px',
          border: '1px solid #f5c6cb',
          borderRadius: '5px'
        }}>
          {alert}
        </div>
      )}

      {/* Video feed toggle button */}
      <button onClick={toggleVideo}>
        {showVideo ? 'Stop Video Feed' : 'Start Video Feed'}
      </button>

      {/* Conditionally render the video feed */}
      {showVideo && (
        <div>
          <h2>Live Video Feed</h2>
          <img
            src="http://localhost:5000/video_feed" // The Flask video stream route
            alt="Video Feed"
            width="640"
            height="480"
          />
        </div>
      )}
    </div>
  );
}

export default App;

"""
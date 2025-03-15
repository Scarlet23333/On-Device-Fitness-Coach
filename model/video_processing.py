from utils.math_utils import calculate_angle, calculate_distance

import eventlet
eventlet.monkey_patch()

from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Exercise state variables
current_exercise = "squat"  # Default exercise
exercise_counts = {
    "squat": 0,
    "pushup": 0,
    "plank": 0,
    "lunge": 0,
    "shoulder_press": 0
}
exercise_timers = {
    "plank": 0  # For tracking plank duration
}
exercise_states = {
    "squat": {"in_progress": False, "added_count": False},
    "pushup": {"in_progress": False, "added_count": False},
    "plank": {"in_progress": False, "start_time": None},
    "lunge": {"in_progress": False, "added_count": False, "leg": "left"},
    "shoulder_press": {"in_progress": False, "added_count": False}
}

# Thresholds for exercises
thresholds = {
    "squat": {
        "knee_angle_threshold": 100,
        "knee_angle_threshold_low": 60,
        "hip_angle_threshold": 130
    },
    "pushup": {
        "elbow_angle_threshold_high": 160,
        "elbow_angle_threshold_low": 90
    },
    "plank": {
        "body_alignment_threshold": 15,  # Degrees of deviation allowed
        "min_duration": 10  # Minimum seconds for a valid plank
    },
    "lunge": {
        "front_knee_angle_threshold": 110,
        "front_knee_angle_threshold_low": 85,
        "back_knee_angle_threshold": 130,
        "back_knee_angle_threshold_low": 90
    },
    "shoulder_press": {
        "elbow_angle_threshold_high": 160,
        "elbow_angle_threshold_low": 90,
        "shoulder_alignment_threshold": 20  # Degrees
    }
}

def get_landmarks(results):
    """Extract and return landmarks in a more accessible format"""
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    points = {}
    
    # Map important landmarks to points dictionary
    points["nose"] = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, 
                      landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    
    # Left side landmarks
    points["left_shoulder"] = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    points["left_elbow"] = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    points["left_wrist"] = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    points["left_hip"] = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    points["left_knee"] = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                           landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    points["left_ankle"] = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    # Right side landmarks
    points["right_shoulder"] = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    points["right_elbow"] = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    points["right_wrist"] = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    points["right_hip"] = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    points["right_knee"] = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    points["right_ankle"] = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Additional points for better body alignment detection
    points["left_ear"] = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
    points["right_ear"] = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, 
                           landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
    
    # Calculate visibility to determine which side to use
    left_visibility = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility +
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility +
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility) / 3
    
    right_visibility = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility +
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility +
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility) / 3
    
    points["preferred_side"] = "left" if left_visibility > right_visibility else "right"
    
    return points

def detect_squat(points, frame):
    """Detect and count squats"""
    state = exercise_states["squat"]
    thrs = thresholds["squat"]
    
    # Calculate joint angles
    knee_angle = calculate_angle(points["left_hip"], points["left_knee"], points["left_ankle"])
    hip_angle = calculate_angle(points["left_knee"], points["left_hip"], points["left_shoulder"])
    
    # Display angles on frame
    cv2.putText(frame, f'Knee Angle: {int(knee_angle)} degrees', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Hip Angle: {int(hip_angle)} degrees', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    feedback = None
    
    # Detect squat movement
    if hip_angle < thrs["hip_angle_threshold"] and not state["in_progress"]:
        state["in_progress"] = True
    elif hip_angle >= thrs["hip_angle_threshold"] and state["in_progress"]:
        state["in_progress"] = False
        if not state["added_count"]:
            feedback = "Your knees were not bending enough!"
        state["added_count"] = False
    
    if state["in_progress"]:
        if knee_angle < thrs["knee_angle_threshold"] and not state["added_count"]:
            exercise_counts["squat"] += 1
            state["added_count"] = True
            feedback = "Good squat!"
        if knee_angle < thrs["knee_angle_threshold_low"]:
            feedback = "Your squat is too deep!"
    
    return feedback

def detect_pushup(points, frame):
    """Detect and count push-ups"""
    state = exercise_states["pushup"]
    thrs = thresholds["pushup"]
    
    # Use the side with better visibility
    side = points["preferred_side"]
    
    # Calculate elbow angle
    if side == "left":
        elbow_angle = calculate_angle(points["left_shoulder"], points["left_elbow"], points["left_wrist"])
    else:
        elbow_angle = calculate_angle(points["right_shoulder"], points["right_elbow"], points["right_wrist"])
    
    # Display angle on frame
    cv2.putText(frame, f'Elbow Angle: {int(elbow_angle)} degrees', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    feedback = None
    
    # Detect push-up movement
    if elbow_angle < thrs["elbow_angle_threshold_high"] and not state["in_progress"]:
        state["in_progress"] = True
    elif elbow_angle >= thrs["elbow_angle_threshold_high"] and state["in_progress"]:
        state["in_progress"] = False
        if state["added_count"]:
            feedback = "Good push-up!"
        state["added_count"] = False
    
    if state["in_progress"]:
        if elbow_angle <= thrs["elbow_angle_threshold_low"] and not state["added_count"]:
            exercise_counts["pushup"] += 1
            state["added_count"] = True
        elif elbow_angle > thrs["elbow_angle_threshold_low"] and elbow_angle < 130 and not state["added_count"]:
            feedback = "Go lower! Bend your elbows more."
    
    return feedback

def detect_plank(points, frame):
    """Detect and time planks"""
    state = exercise_states["plank"]
    thrs = thresholds["plank"]
    
    # Calculate body alignment angles
    shoulder_hip_ankle_angle = calculate_angle(points["left_shoulder"], points["left_hip"], points["left_ankle"])
    hip_shoulder_ankle_angle = calculate_angle(points["left_hip"], points["left_shoulder"], points["left_ankle"])
    
    # Display angles on frame
    cv2.putText(frame, f'Alignment: {int(shoulder_hip_ankle_angle)} deg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    current_time = time.time()
    feedback = None
    
    # Good plank alignment: body should be straight
    is_aligned = abs(180 - shoulder_hip_ankle_angle) < thrs["body_alignment_threshold"]
    
    # Start plank timer
    if is_aligned and not state["in_progress"]:
        state["in_progress"] = True
        state["start_time"] = current_time
        feedback = "Plank position detected! Hold it steady."
    
    # End plank timer if alignment is lost
    elif not is_aligned and state["in_progress"]:
        duration = current_time - state["start_time"]
        state["in_progress"] = False
        
        if duration >= thrs["min_duration"]:
            exercise_timers["plank"] += duration
            feedback = f"Good plank! Held for {duration:.1f} seconds."
        else:
            feedback = f"Plank too short ({duration:.1f} sec). Try to hold at least {thrs['min_duration']} seconds."
    
    # Provide feedback during plank
    elif is_aligned and state["in_progress"]:
        duration = current_time - state["start_time"]
        # Display timer on frame
        cv2.putText(frame, f'Plank time: {duration:.1f}s', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Periodic encouragement every 5 seconds
        if int(duration) % 5 == 0 and int(duration) > 0:
            feedback = f"Great job! Keep holding for {int(duration)} seconds so far."
    
    # Display total plank time
    cv2.putText(frame, f'Total plank time: {exercise_timers["plank"]:.1f}s', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    return feedback

def detect_lunge(points, frame):
    """Detect and count lunges"""
    state = exercise_states["lunge"]
    thrs = thresholds["lunge"]
    leg = state["leg"]  # Current leg being tracked
    
    if leg == "left":
        front_knee_angle = calculate_angle(points["left_hip"], points["left_knee"], points["left_ankle"])
        back_knee_angle = calculate_angle(points["right_hip"], points["right_knee"], points["right_ankle"])
    else:
        front_knee_angle = calculate_angle(points["right_hip"], points["right_knee"], points["right_ankle"])
        back_knee_angle = calculate_angle(points["left_hip"], points["left_knee"], points["left_ankle"])
    
    # Display angles on frame
    cv2.putText(frame, f'Front Knee: {int(front_knee_angle)} deg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Back Knee: {int(back_knee_angle)} deg', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Tracking {leg} leg', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    feedback = None
    
    # Detect lunge movement
    front_knee_good = (thrs["front_knee_angle_threshold_low"] <= front_knee_angle <= 
                       thrs["front_knee_angle_threshold"])
    back_knee_good = (thrs["back_knee_angle_threshold_low"] <= back_knee_angle <= 
                     thrs["back_knee_angle_threshold"])
    
    # Start of lunge
    if front_knee_good and back_knee_good and not state["in_progress"]:
        state["in_progress"] = True
        feedback = f"Good lunge position with {leg} leg forward!"
    
    # End of lunge (back to standing)
    elif (front_knee_angle > thrs["front_knee_angle_threshold"] and 
          back_knee_angle > thrs["back_knee_angle_threshold"] and 
          state["in_progress"]):
        state["in_progress"] = False
        if not state["added_count"]:
            feedback = f"Finish your lunge movement by bending both knees more."
        state["added_count"] = False
    
    # During lunge - count when both knees are at proper angles
    if state["in_progress"]:
        if front_knee_good and back_knee_good and not state["added_count"]:
            exercise_counts["lunge"] += 1
            state["added_count"] = True
            feedback = f"Good lunge with {leg} leg forward!"
        
        # Form feedback
        elif front_knee_angle < thrs["front_knee_angle_threshold_low"] and not state["added_count"]:
            feedback = "Front knee is bending too much!"
        elif back_knee_angle < thrs["back_knee_angle_threshold_low"] and not state["added_count"]:
            feedback = "Back knee is bending too much!"
    
    # Switch legs after every 5 lunges
    if exercise_counts["lunge"] > 0 and exercise_counts["lunge"] % 5 == 0 and not state["in_progress"]:
        state["leg"] = "right" if leg == "left" else "left"
        feedback = f"Great! Now switch to {state['leg']} leg forward."
    
    return feedback

def detect_shoulder_press(points, frame):
    """Detect and count shoulder presses"""
    state = exercise_states["shoulder_press"]
    thrs = thresholds["shoulder_press"]
    
    # Calculate arm angles
    left_elbow_angle = calculate_angle(points["left_shoulder"], points["left_elbow"], points["left_wrist"])
    right_elbow_angle = calculate_angle(points["right_shoulder"], points["right_elbow"], points["right_wrist"])
    
    # Calculate shoulder alignment
    left_shoulder_angle = calculate_angle(points["left_hip"], points["left_shoulder"], points["left_elbow"])
    right_shoulder_angle = calculate_angle(points["right_hip"], points["right_shoulder"], points["right_elbow"])
    
    # Use average of both arms
    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
    
    # Display angles on frame
    cv2.putText(frame, f'Elbow Angle: {int(avg_elbow_angle)} deg', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'L Shoulder: {int(left_shoulder_angle)} deg', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'R Shoulder: {int(right_shoulder_angle)} deg', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    feedback = None
    
    # Check if shoulders are correctly positioned
    shoulder_aligned = (abs(90 - left_shoulder_angle) < thrs["shoulder_alignment_threshold"] and 
                        abs(90 - right_shoulder_angle) < thrs["shoulder_alignment_threshold"])
    
    # Detect shoulder press movement
    if avg_elbow_angle < thrs["elbow_angle_threshold_high"] and not state["in_progress"]:
        state["in_progress"] = True
    elif avg_elbow_angle >= thrs["elbow_angle_threshold_high"] and state["in_progress"]:
        state["in_progress"] = False
        if state["added_count"]:
            feedback = "Good shoulder press!"
        state["added_count"] = False
    
    if state["in_progress"]:
        if avg_elbow_angle <= thrs["elbow_angle_threshold_low"] and not state["added_count"]:
            if shoulder_aligned:
                exercise_counts["shoulder_press"] += 1
                state["added_count"] = True
                feedback = "Good shoulder press form!"
            else:
                feedback = "Keep your elbows in line with your shoulders!"
    
    return feedback

def generate_frames():
    """Generate video frames with exercise detection and feedback"""
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for a selfie-view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get landmarks in a more usable format
            points = get_landmarks(results)
            if points:
                feedback = None
                
                # Detect the selected exercise
                if current_exercise == "squat":
                    feedback = detect_squat(points, frame)
                elif current_exercise == "pushup":
                    feedback = detect_pushup(points, frame)
                elif current_exercise == "plank":
                    feedback = detect_plank(points, frame)
                elif current_exercise == "lunge":
                    feedback = detect_lunge(points, frame)
                elif current_exercise == "shoulder_press":
                    feedback = detect_shoulder_press(points, frame)
                
                # Display exercise counts
                y_pos = 120
                for exercise, count in exercise_counts.items():
                    if exercise == "plank":
                        continue  # Plank is displayed separately with time
                    cv2.putText(frame, f'{exercise.capitalize()}: {count}', (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    y_pos += 30
                
                # Send feedback to client if available
                if feedback:
                    socketio.emit('posture_alert', {'message': feedback})
        
        # Show current exercise
        cv2.putText(frame, f'Current exercise: {current_exercise.capitalize()}', 
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('change_exercise')
def handle_change_exercise(data):
    """Handle exercise change requests from the client"""
    global current_exercise
    if 'exercise' in data and data['exercise'] in exercise_counts:
        current_exercise = data['exercise']
        socketio.emit('exercise_changed', {'exercise': current_exercise})

@socketio.on('reset_counters')
def handle_reset_counters():
    """Reset all exercise counters"""
    for exercise in exercise_counts:
        exercise_counts[exercise] = 0
    exercise_timers["plank"] = 0
    socketio.emit('counters_reset')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    socketio.emit('exercise_stats', {
        'counts': exercise_counts,
        'timers': exercise_timers,
        'current': current_exercise
    })

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)

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

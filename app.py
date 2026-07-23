from flask import Flask, render_template, Response, request, jsonify, url_for
import cv2
import mediapipe as mp
import math
# import matplotlib.pyplot as plt  # Not used in Flask app
import platform
import subprocess
import time
from threading import Lock
import json
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'yoga_posture_detection_secret_key'  # Required for session management

# Camera state shared across requests
camera_state = {
    'index': 0,
    'capture': None,
    'needs_reopen': True
}
camera_lock = Lock()

CONFIDENCE_THRESHOLD = 60.0
ENABLE_ANGLE_OVERLAY = True
ENABLE_CONFIDENCE_OVERLAY = True
TRAINING_RESULTS_PATH = Path('training_analysis_results.json')
latest_pose_data = {
    'label': 'Unknown Pose',
    'confidence': 0.0,
    'angles': {},
    'joint_confidences': {},
    'timestamp': None
}

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose

# Setting up the Pose function
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 


def load_pose_thresholds():
    """Load pose thresholds from JSON analysis results."""
    if not TRAINING_RESULTS_PATH.exists():
        app.logger.warning("Training analysis results not found; using default thresholds.")
        return None

    try:
        with TRAINING_RESULTS_PATH.open('r', encoding='utf-8') as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        app.logger.warning("Failed to parse training analysis results: %s", exc)
        return None

    thresholds = {}
    for pose_name, pose_data in data.items():
        pose_angles = pose_data.get('angles', {})
        thresholds[pose_name] = {}
        for joint, stats in pose_angles.items():
            thresholds[pose_name][joint] = {
                'min': stats.get('min'),
                'max': stats.get('max'),
                'mean': stats.get('mean'),
                'std': stats.get('std') if stats.get('std') is not None else 0.0
            }
    return thresholds


def get_default_thresholds():
    """Return hardcoded fallback thresholds matching legacy ranges."""
    return {
        'T_Pose': {
            'left_elbow': {'min': 130.0, 'max': 180.0, 'mean': 155.0, 'std': 15.0},
            'right_elbow': {'min': 175.0, 'max': 220.0, 'mean': 197.5, 'std': 15.0},
            'left_shoulder': {'min': 100.0, 'max': 200.0, 'mean': 150.0, 'std': 25.0},
            'right_shoulder': {'min': 50.0, 'max': 130.0, 'mean': 90.0, 'std': 20.0},
            'left_knee': {'min': 165.0, 'max': 195.0, 'mean': 180.0, 'std': 10.0},
            'right_knee': {'min': 165.0, 'max': 195.0, 'mean': 180.0, 'std': 10.0}
        },
        'Tree_Pose': {
            'left_elbow': {'min': 120.0, 'max': 200.0, 'mean': 160.0, 'std': 20.0},
            'right_elbow': {'min': 180.0, 'max': 230.0, 'mean': 205.0, 'std': 15.0},
            'left_shoulder': {'min': 150.0, 'max': 210.0, 'mean': 180.0, 'std': 15.0},
            'right_shoulder': {'min': 150.0, 'max': 210.0, 'mean': 180.0, 'std': 15.0},
            'left_knee': {'min': 165.0, 'max': 195.0, 'mean': 180.0, 'std': 10.0},
            'right_knee': {'min': 25.0, 'max': 45.0, 'mean': 35.0, 'std': 7.0}
        }
    }


POSE_THRESHOLDS = load_pose_thresholds() or get_default_thresholds()
if POSE_THRESHOLDS is not None and TRAINING_RESULTS_PATH.exists():
    app.logger.info("Loaded pose thresholds from training data.")
else:
    app.logger.info("Using default pose thresholds.")

def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                (landmark.z * width)))
    
    
    return output_image,landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle


def calculate_angle_confidence(detected_angle, reference_stats):
    """Calculate confidence score (0-100) for a joint angle."""
    if detected_angle is None or reference_stats is None:
        return 0.0

    mean = reference_stats.get('mean')
    std = reference_stats.get('std', 0.0) or 0.0
    min_angle = reference_stats.get('min')
    max_angle = reference_stats.get('max')

    if mean is None:
        if min_angle is not None and max_angle is not None:
            return 100.0 if min_angle <= detected_angle <= max_angle else 0.0
        return 0.0

    if std == 0.0:
        if min_angle is not None and max_angle is not None:
            tolerance = max((max_angle - min_angle) / 2.0, 1.0)
        else:
            tolerance = 20.0
    else:
        tolerance = max(std * 2.0, 1.0)

    deviation = abs(detected_angle - mean)
    confidence = 100.0 - (deviation / tolerance) * 100.0
    return max(0.0, min(confidence, 100.0))


def calculate_pose_confidence(pose_name, detected_angles, thresholds):
    """Aggregate joint confidences into a pose confidence."""
    pose_thresholds = thresholds.get(pose_name, {})
    joint_confidences = {}

    for joint, angle in detected_angles.items():
        if joint not in pose_thresholds:
            continue
        joint_confidences[joint] = calculate_angle_confidence(angle, pose_thresholds[joint])

    if not joint_confidences:
        return 0.0, joint_confidences

    if pose_name == 'Tree_Pose':
        weights = {
            'left_knee': 0.4,
            'right_knee': 0.4,
            'left_shoulder': 0.1,
            'right_shoulder': 0.1,
            'left_elbow': 0.1,
            'right_elbow': 0.1,
        }
    else:
        weights = {joint: 1.0 for joint in joint_confidences.keys()}

    weighted_sum = 0.0
    total_weight = 0.0
    for joint, confidence in joint_confidences.items():
        weight = weights.get(joint, 1.0)
        if weight == 0.0:
            continue
        weighted_sum += confidence * weight
        total_weight += weight

    if total_weight == 0.0:
        overall_confidence = sum(joint_confidences.values()) / len(joint_confidences)
    else:
        overall_confidence = weighted_sum / total_weight

    return max(0.0, min(overall_confidence, 100.0)), joint_confidences


def classifyPose(landmarks, output_image, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    confidence = 0.0
    joint_confidences = {}

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Get the angle between the left hip, knee and ankle points. 
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right hip, knee and ankle points 
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #----------------------------------------------------------------------------------------------------------------
    
    detected_angles = {
        'left_elbow': left_elbow_angle,
        'right_elbow': right_elbow_angle,
        'left_shoulder': left_shoulder_angle,
        'right_shoulder': right_shoulder_angle,
        'left_knee': left_knee_angle,
        'right_knee': right_knee_angle,
    }

    t_pose_thresholds = POSE_THRESHOLDS.get('T_Pose', {})
    tree_pose_thresholds = POSE_THRESHOLDS.get('Tree_Pose', {})

    def angle_in_stats(angle_value, stats):
        if angle_value is None or stats is None:
            return False
        min_angle = stats.get('min')
        max_angle = stats.get('max')
        if min_angle is not None and angle_value < min_angle:
            return False
        if max_angle is not None and angle_value > max_angle:
            return False
        return True

    def within_threshold(joint_name, thresholds):
        return angle_in_stats(detected_angles.get(joint_name), thresholds.get(joint_name))

    required_t_pose_joints = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']
    if all(within_threshold(joint, t_pose_thresholds) for joint in required_t_pose_joints):
        label = 'T Pose'
        confidence, joint_confidences = calculate_pose_confidence('T_Pose', detected_angles, POSE_THRESHOLDS)
        if confidence < CONFIDENCE_THRESHOLD:
            label = 'Unknown Pose'
            confidence = 0.0
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if the both arms are straight.
    if label == 'Unknown Pose' and left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:

        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:

    # Check if it is the warrior II pose.
    #----------------------------------------------------------------------------------------------------------------

            # Check if one leg is straight.
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:

                # Check if the other leg is bended at the required angle.
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:

                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose' 
                        
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the T pose.
    #----------------------------------------------------------------------------------------------------------------
    
            # Check if both legs are straight
            # if left_knee_angle > 160 and left_knee_angle < 195 and right_knee_angle > 160 and right_knee_angle < 195:

            #     # Specify the label of the pose that is tree pose.
            #     label = 'T Pose'

    #----------------------------------------------------------------------------------------------------------------
    
    # Check if it is the tree pose.
    #----------------------------------------------------------------------------------------------------------------
    
    # Check if one leg is straight
    if label == 'Unknown Pose':
        upper_body_ok = all(within_threshold(joint, tree_pose_thresholds) for joint in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow'])

        left_knee_stats = tree_pose_thresholds.get('left_knee')
        right_knee_stats = tree_pose_thresholds.get('right_knee')

        tree_standard = angle_in_stats(left_knee_angle, left_knee_stats) and angle_in_stats(right_knee_angle, right_knee_stats)
        tree_mirrored = angle_in_stats(left_knee_angle, right_knee_stats) and angle_in_stats(right_knee_angle, left_knee_stats)

        if upper_body_ok and (tree_standard or tree_mirrored):
            label = 'Tree Pose'
            confidence, joint_confidences = calculate_pose_confidence('Tree_Pose', detected_angles, POSE_THRESHOLDS)
            if confidence < CONFIDENCE_THRESHOLD:
                label = 'Unknown Pose'
                confidence = 0.0

    #----------------------------------------------------------------------------------------------------------------
    
   
   
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        pose_key = label.replace(' ', '_') if label else ''
        if pose_key in POSE_THRESHOLDS:
            if confidence >= 85.0:
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)
        else:
            color = (0, 255, 0)
    
    # Write the label on the output image. 
    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    if ENABLE_CONFIDENCE_OVERLAY and label != 'Unknown Pose':
        confidence_color = (0, 255, 0)
        if confidence < 80:
            confidence_color = (0, 255, 255)
        if confidence < CONFIDENCE_THRESHOLD:
            confidence_color = (0, 165, 255)
        cv2.putText(output_image, f'Confidence: {confidence:.1f}%', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, confidence_color, 2)

    if ENABLE_ANGLE_OVERLAY and label != 'Unknown Pose':
        sorted_joints = sorted(joint_confidences.items(), key=lambda item: item[1], reverse=True)
        for idx, (joint, joint_conf) in enumerate(sorted_joints[:3]):
            angle_value = detected_angles.get(joint)
            if angle_value is None:
                continue
            text = f"{joint.replace('_', ' ').title()}: {angle_value:.1f}° ({joint_conf:.0f}%)"
            cv2.putText(output_image, text, (10, 110 + idx * 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    
    # Check if the resultant image is specified to be displayed.
    # if display:
    #     # Display the resultant image.
    #     plt.figure(figsize=[10,10])
    #     plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');

    return output_image, label, confidence, detected_angles

def get_windows_camera_names():
    """Get camera names from Windows registry and devices"""
    if platform.system() != 'Windows':
        return {}

    camera_names = {}
    try:
        # Get video capture devices from registry
        result = subprocess.check_output(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | Select-Object -ExpandProperty Name",
            ],
            encoding='utf-8',
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        names = [line.strip() for line in result.splitlines() if line.strip()]
        
        # Map names to indices (best effort)
        for idx, name in enumerate(names):
            camera_names[idx] = name
            
    except Exception:
        pass
    
    return camera_names


def _create_capture(index: int):
    """Create a video capture object for the given camera index"""
    cap = None
    
    if platform.system() == 'Windows':
        # Try DirectShow first (better for most cameras)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        # If DirectShow fails, try default backend (works better for virtual cameras like DroidCam)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)

    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None

    return cap


def _open_camera(index: int):
    cap = _create_capture(index)
    if cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1380)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    return cap


def _get_active_capture():
    with camera_lock:
        if camera_state['needs_reopen'] or camera_state['capture'] is None:
            if camera_state['capture'] is not None:
                camera_state['capture'].release()
                camera_state['capture'] = None

            camera_state['capture'] = _open_camera(camera_state['index'])
            camera_state['needs_reopen'] = camera_state['capture'] is None

        return camera_state['capture']


def get_available_cameras_with_names():
    """Detect all available cameras including virtual cameras like DroidCam"""
    available_cameras = []
    friendly_names = get_windows_camera_names() if platform.system() == 'Windows' else {}

    with camera_lock:
        active_index = camera_state['index']
        active_capture = camera_state['capture'] if not camera_state['needs_reopen'] else None

    # Check more camera indices to catch virtual cameras
    for i in range(20):  # Increased from 10 to 20 to catch more virtual cameras
        cap = None
        release_after = False

        # Reuse active capture if checking the same index
        if active_capture is not None and i == active_index:
            cap = active_capture
        else:
            cap = _create_capture(i)
            release_after = cap is not None

        if not cap:
            continue

        # Try to read a frame to verify camera works
        ret, frame = cap.read()
        if ret and frame is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            backend = cap.getBackendName()
            
            # Get friendly name from system or generate descriptive name
            name = friendly_names.get(i, None)
            if not name:
                # Try to detect DroidCam or other virtual cameras by resolution or other properties
                if 'DroidCam' in backend or width == 1920 or width == 1280:
                    name = f"DroidCam / Virtual Camera {i}"
                elif i == 0:
                    name = f"Integrated Camera {i}"
                else:
                    name = f"Camera {i}"

            available_cameras.append({
                'index': i,
                'name': name,
                'backend': backend,
                'resolution': f"{width}x{height}",
            })

        if release_after and cap:
            cap.release()

    return available_cameras


def _serialize_pose_metrics():
    """Return a JSON-serializable view of the latest pose metrics."""
    data = latest_pose_data.copy()
    timestamp = data.get('timestamp')
    now = time.time()
    elapsed = None

    if timestamp is not None:
        elapsed = max(0.0, now - timestamp)

    angles = {
        joint: round(value, 1)
        for joint, value in (data.get('angles') or {}).items()
        if isinstance(value, (int, float))
    }

    joint_confidences = {
        joint: round(value, 1)
        for joint, value in (data.get('joint_confidences') or {}).items()
        if isinstance(value, (int, float))
    }

    return {
        'label': data.get('label', 'Unknown Pose'),
        'confidence': round(float(data.get('confidence', 0.0)), 1),
        'angles': angles,
        'joint_confidences': joint_confidences,
        'updatedAgo': elapsed,
        'isStale': elapsed is None or elapsed > 3.0
    }


@app.route('/pose_metrics')
def pose_metrics():
    """Expose the most recent pose metrics for the live detection page."""
    return jsonify(_serialize_pose_metrics())


def webcam_feed():
    global latest_pose_data
    try:
        while True:
            capture = _get_active_capture()

            if capture is None:
                time.sleep(0.5)
                continue

            ok, frame = capture.read()

            if not ok:
                with camera_lock:
                    camera_state['needs_reopen'] = True
                time.sleep(0.1)
                continue

            # Flip the frame horizontally for natural (selfie-view) visualization
            frame = cv2.flip(frame, 1)

            # Get the width and height of the frame
            frame_height, frame_width, _ = frame.shape

            # Resize the frame while keeping the aspect ratio
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            # Perform Pose landmark detection
            frame, landmarks = detectPose(frame, pose, display=False)

            if landmarks:
                # Perform the Pose Classification
                frame, label, pose_confidence, angles = classifyPose(landmarks, frame, display=False)
                joint_confidences = {}
                pose_key = label.replace(' ', '_') if label else ''
                if pose_key in POSE_THRESHOLDS:
                    _, joint_confidences = calculate_pose_confidence(pose_key, angles, POSE_THRESHOLDS)
                latest_pose_data = {
                    'label': label,
                    'confidence': pose_confidence,
                    'angles': angles,
                    'joint_confidences': joint_confidences,
                    'timestamp': time.time()
                }
            else:
                latest_pose_data = {
                    'label': 'Unknown Pose',
                    'confidence': 0.0,
                    'angles': {},
                    'joint_confidences': {},
                    'timestamp': time.time()
                }

            # Convert the frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
    except GeneratorExit:
        pass

PAIN_TO_POSES = {
    'neck': ['Sukhasana', 'Balasana', 'Shavasana'],
    'shoulders': ['Trikonasana', 'Virabadrasana', 'Chakrasana'],
    'upper_back': ['Trikonasana', 'Bhujangasana', 'Balasana'],
    'lower_back': ['Balasana', 'Bhujangasana', 'Chakrasana'],
    'hips': ['Trikonasana', 'Virabadrasana', 'Vrikshasana'],
    'knees': ['Virabadrasana', 'Vrikshasana'],
    'ankles': ['Vrikshasana'],
    'stress': ['Sukhasana', 'Balasana', 'Shavasana'],
    'hamstrings': ['Trikonasana', 'Virabadrasana'],
    'calves': ['Virabadrasana'],
    'chest': ['Bhujangasana', 'Chakrasana'],
    'balance': ['Vrikshasana']
}


@app.context_processor
def inject_navigation():
    return {
        'nav_links': [
            {'label': 'Home', 'url': url_for('home')},
            {'label': 'All Poses', 'url': url_for('poses')},
            {'label': 'Try It Out', 'url': url_for('yoga_try')}
        ]
    }


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/poses')
def poses():
    pain_query = request.args.get('pain', '')
    selected_pain_areas = [area.strip().lower() for area in pain_query.split(',') if area.strip()]

    recommended_poses = []
    seen = set()
    for pain_area in selected_pain_areas:
        for pose_name in PAIN_TO_POSES.get(pain_area, []):
            if pose_name not in seen:
                recommended_poses.append(pose_name)
                seen.add(pose_name)

    return render_template(
        'index.html',
        selected_pain_areas=selected_pain_areas,
        recommended_poses=recommended_poses
    )

@app.route('/yoga_try')
def yoga_try():
    return render_template('yoga_try.html')

@app.route('/set_camera', methods=['POST'])
def set_camera():
    camera_index = int(request.form.get('camera_index', camera_state['index']))

    with camera_lock:
        camera_state['index'] = camera_index

        if camera_state['capture'] is not None:
            camera_state['capture'].release()
            camera_state['capture'] = None

        camera_state['needs_reopen'] = True

    return jsonify({'success': True, 'camera_index': camera_index})

@app.route('/get_available_cameras')
def get_available_cameras():
    cameras = get_available_cameras_with_names()

    with camera_lock:
        active_index = camera_state['index']

    return jsonify({'cameras': cameras, 'activeCameraIndex': active_index})

@app.route('/video_feed1')
def video_feed1():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

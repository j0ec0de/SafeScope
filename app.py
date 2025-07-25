import os
import cv2
import time
import subprocess
import pygame
from flask import Flask, request, render_template, send_from_directory, url_for
from inference_sdk import InferenceHTTPClient
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Roboflow API configuration
API_URL = "https://detect.roboflow.com"
API_KEYS = {
    "fire": "iFxN9ERCxxe3jqrTyx9v",
    "weapon": "NSpQ9E0F9NOR9QRirW0F",
    "person": "NSpQ9E0F9NOR9QRirW0F",
    "license_plate": "NSpQ9E0F9NOR9QRirW0F"
}

MODEL_IDS = {
    "fire": "fire-detection-using-yolov5/4",
    "weapon": "weapon_detection-leyfd/5",
    "person": "people-detection-o4rdr/7",
    "license_plate": "platesv2-0nqdl/6"
}

CLIENTS = {key: InferenceHTTPClient(api_url=API_URL, api_key=API_KEYS[key]) for key in API_KEYS}
MIN_CONFIDENCE = 0.7

# 🚨 Alarm Setup
pygame.mixer.init()
ALARM_SOUND = "F:/Projects/Safecope-final/secur-cam-final/alarm2.wav"

def play_alarm():
    """Plays an alarm sound when fire or weapon is detected."""
    pygame.mixer.music.load(ALARM_SOUND)
    pygame.mixer.music.play()

def detect_object(image_path, object_type):
    try:
        return CLIENTS[object_type].infer(image_path, model_id=MODEL_IDS[object_type])
    except Exception as e:
        print(f"Error in {object_type} detection: {e}")
        return {'predictions': []}

def parallel_detect_objects(image_path):
    """Runs all detections in parallel for faster processing."""
    with ThreadPoolExecutor() as executor:
        future_results = {obj: executor.submit(detect_object, image_path, obj) for obj in API_KEYS}
        return {obj: future_results[obj].result() for obj in API_KEYS}

def should_detect_fire(results_dict):
    return not (len(results_dict['license_plate']['predictions']) > 0)

def annotate_image(image_path, results_dict):
    image = cv2.imread(image_path)
    COLORS = {
        "fire": (0, 0, 255),
        "weapon": (0, 255, 0),
        "person": (255, 0, 0),
        "license_plate": (255, 255, 0)
    }
    
    for object_type, results in results_dict.items():
        for detection in results['predictions']:
            if detection['confidence'] >= MIN_CONFIDENCE:
                x, y, w, h = get_coordinates(detection)
                cv2.rectangle(image, (x, y), (x + w, y + h), COLORS[object_type], 2)
                cv2.putText(image, f"{object_type.capitalize()}: {detection['confidence']:.2f}", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[object_type], 2)

    output_image_path = os.path.join(OUTPUT_FOLDER, 'annotated_' + os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)
    return output_image_path

def get_coordinates(detection):
    x_center, y_center, w, h = int(detection['x']), int(detection['y']), int(detection['width']), int(detection['height'])
    return x_center - w // 2, y_center - h // 2, w, h

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = base_name + "_out.mp4"
    temp_output_path = os.path.join(OUTPUT_FOLDER, "temp_" + output_filename)
    final_output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    if not cap.isOpened():
        print("Error: Video file could not be opened!")
        return None, {}

    detections = {"fire": False, "weapon": False, "person": False, "license_plate": False}
    frame_count = 0
    process_every_n_frames = 30  # 🔥 Increased frame skipping

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % process_every_n_frames != 0:
            out.write(frame)
            continue  

        frame_path = os.path.join(UPLOAD_FOLDER, 'temp_frame.jpg')
        cv2.imwrite(frame_path, frame)

        results_dict = parallel_detect_objects(frame_path)

        if not should_detect_fire(results_dict):
            results_dict['fire']['predictions'] = []

        for obj in detections.keys():
            if len(results_dict[obj]['predictions']) > 0:
                detections[obj] = True

        if detections["fire"] or detections["weapon"]:
            play_alarm()  

        annotated_frame_path = annotate_image(frame_path, results_dict)
        annotated_frame = cv2.imread(annotated_frame_path)
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    ffmpeg_command = [
        "ffmpeg", "-y", "-i", temp_output_path, "-c:v", "libx264", "-preset", "superfast",
        "-movflags", "faststart", "-pix_fmt", "yuv420p", final_output_path
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if os.path.exists(final_output_path):
        os.remove(temp_output_path)
        return output_filename, detections
    else:
        return None, {}

def process_image(image_path):
    results_dict = parallel_detect_objects(image_path)

    if not should_detect_fire(results_dict):
        results_dict['fire']['predictions'] = []

    fire_detected = len(results_dict['fire']['predictions']) > 0
    weapon_detected = len(results_dict['weapon']['predictions']) > 0
    person_detected = len(results_dict['person']['predictions']) > 0
    license_plate_detected = len(results_dict['license_plate']['predictions']) > 0

    if fire_detected or weapon_detected:
        play_alarm()

    annotated_image_path = annotate_image(image_path, results_dict)

    return annotated_image_path, fire_detected, weapon_detected, person_detected, license_plate_detected

@app.route('/')
def index():
    return render_template('indexSample.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        output_filename, detections = process_video(file_path)
        return render_template('resultSample.html', 
                               video_url=url_for('serve_video', filename=output_filename), 
                               fire_detected=detections["fire"],
                               weapon_detected=detections["weapon"],
                               person_detected=detections["person"],
                               license_plate_detected=detections["license_plate"])

    elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        annotated_image_path, fire_detected, weapon_detected, person_detected, license_plate_detected = process_image(file_path)
        return render_template('resultSample.html', 
                               image_url=url_for('serve_video', filename=os.path.basename(annotated_image_path)), 
                               fire_detected=fire_detected,
                               weapon_detected=weapon_detected,
                               person_detected=person_detected,
                               license_plate_detected=license_plate_detected)

    return "Unsupported file format", 400

@app.route('/outputs/<filename>')
def serve_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)


    

import os
import cv2
import json
from ultralytics import YOLO
import numpy as np
import config
from scipy.optimize import linear_sum_assignment
import torch
from filterpy.kalman import KalmanFilter
import time
# Use GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: Tuples representing bounding boxes (x, y, w, h)

    Returns:
        float: IoU value
    """
    # Calculate IoU between two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def create_kalman_filter(initial_state):
    """
    Create and initialize a Kalman Filter for tracking an object.
    Args:
        initial_state (tuple): Initial state of the object (x, y, dx, dy).
    Returns:
        KalmanFilter: Initialized Kalman Filter.
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state variables (x, y, dx, dy) and 2 measurements (x, y)
    kf.x = initial_state  # Initial state vector [x, y, dx, dy]
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                     [0, 1, 0, 0]])
    kf.P *= 1000  # Initial uncertainty
    kf.R = np.eye(2) * 10  # Measurement noise
    kf.Q = np.eye(4) * 0.01  # Process noise
    return kf

def update_kalman_filters(kalman_filters, detections):
    """
    Update the Kalman Filters with current detections.
    Args:
        kalman_filters (list): List of KalmanFilter objects.
        detections (list): List of current detections (bounding boxes).
    Returns:
        list: Updated Kalman Filters.
    """
    for kf, detection in zip(kalman_filters, detections):
        measurement = np.array([detection['box'][0], detection['box'][1]])  # x, y
        kf.predict()
        kf.update(measurement)
    return kalman_filters

def object_tracking1(frame, all_objects, current_detections):
    """
    Track objects using Kalman Filters.
    Args:
        frame (numpy array): The current frame.
        all_objects (list): All objects detected in the video so far.
        current_detections (list): All detections in the current frame.
    Returns:
        list: Updated list of tracked objects.
    """
    frame_height, frame_width = frame.shape[:2]
    unmatched_detections = []

    # Update existing objects with Kalman Filters
    for obj in all_objects:
        kf = obj['kf']
        kf.predict()  # Predict next state
        obj_state = kf.x  # Get predicted state [x, y, dx, dy]
        obj['predicted_box'] = (
            obj_state[0],  # x
            obj_state[1],  # y
            obj['value']['sequence'][-1]['width'] * frame_width / 100,  # width
            obj['value']['sequence'][-1]['height'] * frame_height / 100  # height
        )

    # Assign detections to tracked objects using IoU
    for det in current_detections:
        det_box = det['box']
        best_match = None
        best_iou = 0

        for obj in all_objects:
            iou = calculate_iou(obj['predicted_box'], det_box)
            if iou > best_iou:
                best_match = obj
                best_iou = iou

        if best_match and best_iou > 0.3:  # Match found with sufficient IoU
            measurement = np.array([det_box[0], det_box[1]])
            best_match['kf'].update(measurement)
            best_match['value']['sequence'].append({
                "frame": frame,
                "enabled": True,
                "x": det_box[0] / frame_width * 100,
                "y": det_box[1] / frame_height * 100,
                "width": det_box[2] / frame_width * 100,
                "height": det_box[3] / frame_height * 100,
            })
        else:
            unmatched_detections.append(det)

    # Create new objects for unmatched detections
    for det in unmatched_detections:
        initial_state = np.array([det['box'][0], det['box'][1], 0, 0])  # x, y, dx, dy
        kf = create_kalman_filter(initial_state)
        new_obj = {
            "kf": kf,
            "value": {
                "sequence": [{
                    "frame": frame,
                    "enabled": True,
                    "x": det['box'][0] / frame_width * 100,
                    "y": det['box'][1] / frame_height * 100,
                    "width": det['box'][2] / frame_width * 100,
                    "height": det['box'][3] / frame_height * 100,
                }],
                "labels": [det['class']]
            }
        }
        all_objects.append(new_obj)

    return all_objects


def object_detection1(frames):
    """
    Perform object detection on video frames using YOLOv8.

    This function detects objects in each frame, tracks them across frames,
    and assigns unique IDs to each object.

    Args:
        frames: List of video frames

    Returns:
        list: Detected objects with their bounding boxes and labels
    """
    model = YOLO(config.OBJECT_DETECTION_MODEL)
    all_objects = []
    object_id = 0

    for i, frame in enumerate(frames):
        detections = model.predict(frame, verbose=False)[0]
        current_detections = []

        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = det
            w, h = x2 - x1, y2 - y1
            
            # Chỉ xử lý các đối tượng có class = 0 (người)
            if int(cls) == 0 and conf > config.CONFIDENCE_THRESHOLD:
                current_detections.append({
                    "box": (x1, y1, w, h),
                    "confidence": conf,
                    "class": "person"
                })

        # Update object tracking using Kalman Filter
        all_objects = object_tracking1(frame, all_objects, current_detections)

    return all_objects




def convert_fps(frames, original_fps, target_fps):
    """
    Resample a list of frames from the original FPS to the target FPS.

    The resampling is done by linear interpolation between the two nearest frames.
    If the target frame index is an integer, the frame is taken from the original list.
    If the target frame index is not an integer, the frame is interpolated between the two nearest frames.

    Args:
        frames (list): List of frames
        original_fps (float): Original FPS
        target_fps (float): Target FPS

    Returns:
        list: Resampled list of frames
    """
    # Calculate the ratio between the original and target FPS
    time_ratio = original_fps / target_fps

    # Calculate the number of frames in the target FPS
    target_frame_count = int(len(frames) / time_ratio)

    # Create a list to store the resampled frames
    resampled_frames = []

    for i in range(target_frame_count):
        # Calculate the corresponding frame index in the original list
        original_index = i * time_ratio

        # If it's not an integer index, we need to interpolate
        if not original_index.is_integer():
            lower_index = int(original_index)
            upper_index = min(lower_index + 1, len(frames) - 1)

            # Calculate the weight for interpolation
            weight = original_index - lower_index

            # Interpolate between the two nearest frames
            interpolated_frame = cv2.addWeighted(
                frames[lower_index], 1 - weight,
                frames[upper_index], weight, 0
            )

            resampled_frames.append(interpolated_frame)
        else:
            # If it's an integer index, just take that frame
            resampled_frames.append(frames[int(original_index)])

    return resampled_frames

def draw_bounding_boxes(frames, all_results):
    """
    Draw bounding boxes on the frames based on all_results.

    Args:
        frames (list): List of frames (numpy arrays).
        all_results (list): List of detected and tracked objects.

    Returns:
        list: List of frames with bounding boxes drawn.
    """
    for obj in all_results:
        for state in obj['value']['sequence']:
            # Match frame directly
            frame = state['frame']
            
            # Calculate bounding box coordinates
            x = int(state['x'] * frame.shape[1] / 100)
            y = int(state['y'] * frame.shape[0] / 100)
            w = int(state['width'] * frame.shape[1] / 100)
            h = int(state['height'] * frame.shape[0] / 100)
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, obj['value']['labels'][0], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frames

def handle_video1(video_path):
    """
    Main function to process a video file.

    Args:
        video_path: Path to the input video file
    """
    # Read video and turn it into a frame list
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Resample frames to target FPS and subsample for object detection
    frame_distance = int(original_fps / config.PROCESSING_FPS)
    frames = frames[::frame_distance]

    # Perform object detection
    all_results = object_detection1(frames)
    # Đo thời gian phát video
    display_start_time = time.time()
    # Draw bounding boxes on frames
    frames_with_boxes = draw_bounding_boxes(frames, all_results)

    # Display the video frame by frame
    for frame in frames_with_boxes:
        cv2.imshow("Video with Bounding Boxes", frame)
        if cv2.waitKey(int(1000 / config.PROCESSING_FPS)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    display_end_time = time.time()
    display_duration = display_end_time - display_start_time

    print(f"Thời gian phát video: {display_duration:.4f} giây")

def main():
    # Bắt đầu đo thời gian
    start_time = time.time()
    # Đường dẫn tới video cần xử lý
    video_path = "D:/Dagoras/Fall-Detection/dataset/video/istockphoto-1286618209-640_adpp_is.mp4"

    # Kiểm tra nếu file video tồn tại
    if not os.path.exists(video_path):
        print(f"File video không tồn tại: {video_path}")
        return

    # Gọi hàm xử lý video và phát video với bounding boxes
    handle_video1(video_path)
    # Kết thúc đo thời gian
    end_time = time.time()

    # Tính thời gian xử lý
    execution_time = end_time - start_time
    print(f"Thời gian xử lý: {execution_time:.4f} giây")

# Chỉ chạy phần dưới khi file này được chạy trực tiếp
if __name__ == "__main__":
    main()



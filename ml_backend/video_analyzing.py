import os
import cv2
from ultralytics import YOLO
import numpy as np
import config
from scipy.optimize import linear_sum_assignment
import torch
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


def object_tracking(frame, all_objects, current_detections):
    """
    Track objects across frames by calculating IoU between all detected objects in the current frame
    and the last detection in each object's sequence.

    Args:
        frame (numpy array): The current frame
        all_objects (list): All objects detected in the video so far
        current_detections (list): All detections in the current frame

    Returns:
        tuple: A tuple of two elements.
            - iou_matrix (numpy array): The IoU matrix between all objects and current detections
            - matched_indices (list): A list of tuples, where each tuple contains the index of an object and the index of its best match in the current detections
    """
    iou_matrix = np.zeros((len(all_objects), len(current_detections)))
    for ii, obj in enumerate(all_objects):
        last_detection = obj['value']['sequence'][-1]
        last_box = (
            last_detection['x'] * frame.shape[1] / 100,
            last_detection['y'] * frame.shape[0] / 100,
            last_detection['width'] * frame.shape[1] / 100,
            last_detection['height'] * frame.shape[0] / 100
        )
        for j, det in enumerate(current_detections):
            iou_matrix[ii, j] = calculate_iou(last_box, det['box']) if obj['value']['labels'][0] == det['class'] else 0

    # Use Hungarian algorithm to find best matches
    if len(all_objects) > 0 and len(current_detections) > 0:
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = list(zip(matched_indices[0], matched_indices[1]))
    else:
        matched_indices = []
    return iou_matrix, matched_indices


def object_detection(frames, frame_distance):
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
            if cls == 0:
                current_detections.append({
                    "box": (x1, y1, w, h),
                    "confidence": conf,
                    "class": config.OD_CLASSES[int(cls)]
                })

        iou_matrix, matched_indices = object_tracking(frame, all_objects, current_detections)

        unmatched_detections = list(range(len(current_detections)))
        for obj_idx, det_idx in matched_indices:
            if iou_matrix[obj_idx, det_idx] > 0:  # Only consider matches with IoU > 0
                det = current_detections[det_idx]
                x1, y1, w, h = det['box']
                all_objects[obj_idx]['value']['sequence'].append({
                    "frame": i * frame_distance,
                    "enabled": True,
                    "rotation": 0,
                    "x": x1 / frame.shape[1] * 100,
                    "y": y1 / frame.shape[0] * 100,
                    "width": w / frame.shape[1] * 100,
                    "height": h / frame.shape[0] * 100,
                    "time": i / config.PROCESSING_FPS,
                    "score": det['confidence'],
                })
                unmatched_detections.remove(det_idx)

        # Create new objects for unmatched detections
        for det_idx in unmatched_detections:
            det = current_detections[det_idx]
            x1, y1, w, h = det['box']
            new_obj = {
                "value": {
                    "framesCount": len(frames),
                    "duration": len(frames) / config.PROCESSING_FPS,
                    "sequence": [{
                        "frame": i * frame_distance,
                        "enabled": True,
                        "rotation": 0,
                        "x": x1 / frame.shape[1] * 100,
                        "y": y1 / frame.shape[0] * 100,
                        "width": w / frame.shape[1] * 100,
                        "height": h / frame.shape[0] * 100,
                        "time": i / config.PROCESSING_FPS,
                        "score": det['confidence'],
                    }],
                    "labels": [det['class']]
                },
                "from_name": "bounding_box",
                "to_name": "video",
                "type": "videorectangle",
            }
            all_objects.append(new_obj)
            object_id += 1

        # For objects not detected in this frame, add a disabled detection
        for obj in all_objects:
            if obj['value']['sequence'][-1]['frame'] < i * frame_distance:
                obj['value']['sequence'][-1]['enabled'] = False

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



def handle_video(video_path):
    """
    Main function to process a video file.

    This function coordinates the entire video analysis process, including:
    1. Reading the video and extracting frames
    2. Performing object detection

    Args:
        video_path: Path to the input video file
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    frame_distance = int(original_fps / config.PROCESSING_FPS)
    frames = frames[::frame_distance]

    od_results = object_detection(frames, frame_distance)
    # Đo thời gian phát video
    display_start_time = time.time()
     # Iterate through frames and draw bounding boxes
    for i, frame in enumerate(frames):
        for obj in od_results:
            for seq in obj['value']['sequence']:
                if seq['frame'] == i * frame_distance and seq['enabled']:
                    x = int(seq['x'] / 100 * frame.shape[1])
                    y = int(seq['y'] / 100 * frame.shape[0])
                    w = int(seq['width'] / 100 * frame.shape[1])
                    h = int(seq['height'] / 100 * frame.shape[0])
                    label = obj['value']['labels'][0]
                    score = seq['score']

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Add label and confidence
                    cv2.putText(frame, f"{label}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Object Detection", frame)
        delay = int(1000 / config.PROCESSING_FPS)  # Delay for each frame in ms
        if cv2.waitKey(delay) & 0xFF == ord('q'):
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
    handle_video(video_path)
    # Kết thúc đo thời gian
    end_time = time.time()

    # Tính thời gian xử lý
    execution_time = end_time - start_time
    print(f"Thời gian xử lý: {execution_time:.4f} giây")

# Chỉ chạy phần dưới khi file này được chạy trực tiếp
if __name__ == "__main__":
    main()





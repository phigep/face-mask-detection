import sys 
import os
import time
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2', '1', etc.

# Get the root directory (one level above script directory)
root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Add both to sys.path
sys.path.append(root_dir)  # Add root directory to path
sys.path.append(script_dir)  # Add script directory to path (optional)

import cv2
import numpy as np
import tensorflow as tf
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from xgboost import XGBClassifier
import keras
# Load YOLO model for face detection
yolo_model = YOLO('yolov11l-face.pt')  # Change to your YOLO model
xgb_model = XGBClassifier()
xgb_model.load_model("./training/checkpoints/xgboost_mbconv_model.bin")
# Load TensorFlow model for face classification
from training.models import get_fcn_model,mbconv_base, HybridModel,get_model,effnetv2b0_base
from keras.layers import GlobalAveragePooling2D,Dropout,Dense,Input
from keras.models import Model

def get_hybrid_mlp():

    last_mbconv_layer_name = 'functional_1'  # Example name
    last_mbconv_layer = fcn_modelmbconv.get_layer(last_mbconv_layer_name)

    # Create a new model that outputs up to the last mbconv_block
    inputs = Input(shape=(224,224,3))

    for layer in fcn_modelmbconv.layers:
        layer.trainable = False
    base = last_mbconv_layer(inputs)

    x = GlobalAveragePooling2D()(base)
    x = Dropout(rate=0.3)(x)
    outputs = Dense(2, activation="sigmoid")(x)
    fcn_model = Model(inputs=inputs,outputs = outputs)
    return fcn_model

fcn_modelmbconv = get_fcn_model(mbconv_base,input_dim=(224,224,3), resize_dim=(224,224,3), classes=2)
fcn_modelmbconv.load_weights("./training/checkpoints/fcn_mbconv.weights.h5")

hybrid_mlp_model = get_hybrid_mlp()
hybrid_mlp_model = keras.models.load_model(f"./training/checkpoints/hybridfcnmlp.keras")
enetb0_model = get_model(effnetv2b0_base,input_dim=(224,224,3), resize_dim=(224,224,3), classes=2,classifier_activation="sigmoid", training_base=False)

enetb0_model = keras.models.load_model(f"./training/checkpoints/effnetv2b0frozen.keras")
# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Class labels for classification
class_labels = ['Mask', 'NoMask']  # Replace with actual class names

def preprocess_image(image):
    """Preprocess the image for CNN classification."""
    img = cv2.resize(image, (224, 224))  # Adjust size according to CNN input
    #img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    print(np.shape(img))
    return img

from collections import deque
track_predictions = {}
def smooth_predictions(track_id, prediction, max_frames=10):
    #print("prediction", prediction)
    """Smooth predictions using a rolling buffer."""
    if track_id not in track_predictions:
        track_predictions[track_id] = deque(maxlen=max_frames)
    track_predictions[track_id].append(prediction[0])
    # Calculate the average
    avg_prediction = sum(track_predictions[track_id]) / len(track_predictions[track_id])
    
    # If the average is >= 0.5, classify as 1; otherwise 0
    smoothed_prediction = 1 if avg_prediction >= 0.5 else 0
    return smoothed_prediction


def process_frame(frame):
    """Detect faces, track them and classify."""
    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # Nx6 or Nx7 shape typically
    
    # Ensure correct format: [x1, y1, x2, y2, confidence]
    formatted_detections = []
    for det in detections:
        if len(det) >= 5:
            x1, y1, x2, y2, conf = det[:5]
            formatted_detections.append([x1, y1, x2, y2, conf])
    
    # Now build the list that deep_sort_realtime wants:
    # Each element: [ [x1, y1, x2, y2], confidence, class_id ]
    deep_sort_detections = []
    for (x1, y1, x2, y2, conf) in formatted_detections:
        # Calculate width and height of the original bounding box
        width = x2 - x1
        height = y2 - y1

        # Scale factor to increase by 20%
        scale_factor = 1.8

        # New dimensions after scaling
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Calculate the center of the bounding box
        cx = x1 + width / 2
        cy = y1 + height / 2

        # Compute new bounding box coordinates
        new_x1 = int(cx - new_width / 2)
        new_y1 = int(cy - new_height / 2)
        new_x2 = int(cx + new_width / 2)
        new_y2 = int(cy + new_height / 2)

        # Ensure the new bounding box stays within frame boundaries
        new_x1 = max(new_x1, 0)
        new_y1 = max(new_y1, 0)
        new_x2 = min(new_x2, frame.shape[1] - 1)
        new_y2 = min(new_y2, frame.shape[0] - 1)

        # Convert to DeepSORT format: [x, y, w, h]
        bbox = [new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1]
        deep_sort_detections.append([bbox, conf, 0])
    
    # Update tracks with the properly formatted detections
    tracks = tracker.update_tracks(deep_sort_detections, frame=frame)

    # Then the usual tracking/classification code
    for i,track in enumerate(tracks):
        if track.is_confirmed() and track.time_since_update == 0:
            track_id = track.track_id
            #print(track_id)
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_input = preprocess_image(face_crop)
                prediction_ftrs = fcn_modelmbconv.predict(face_input)
                #print(prediction_ftrs)
                prediction = xgb_model.predict(prediction_ftrs)
                prediction_effnet = np.argmax(enetb0_model.predict(face_input),axis=1)[0]
                prediction_mlp = np.argmax(hybrid_mlp_model.predict(face_input),axis=1)[0]
                print(prediction_effnet)
                #print("predicted ",prediction)
                smoothed_prediction = smooth_predictions(track_id, prediction)
                
                # In XGBoost, 'prediction' might be the predicted class
                # so you may not need np.argmax here, depending on how you trained your model.
                # If 'prediction' is [0 or 1 or 2], use that directly. 
                # If it's a probability distribution, adjust accordingly.
                
                class_index = int(smoothed_prediction) 
                label = class_labels[class_index]
                label_effnet = class_labels[int(prediction_effnet)]
                label_mlp = class_labels[int(prediction_mlp)]
                # If you want a confidence score, you might call predict_proba:
                # confidence = xgb_model.predict_proba(prediction_ftrs)[0][class_index]

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'FCN-XGBoost: {label}', (x1, y1 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f'Effnet: {label_effnet}', (x1, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f'FCN-MLP: {label_mlp}', (x1, y1 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                #time.sleep(1)
    
    return frame

def main(video_source=0, output_file="output_video.mp4"):
    """Main function to process video and save the output."""
    cap = cv2.VideoCapture(video_source)

    # Get video properties for writing output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 file format
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        # Show the processed frame
        cv2.imshow('Face Detection & Classification', processed_frame)

        # Write the processed frame to output video file
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_file}")


if __name__ == "__main__":
    #main('./test_face_mask.mp4')  # Change to your video source
    main("./vid5.mp4",output_file="outvid5.mp4")
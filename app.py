import streamlit as st
import cv2
import numpy as np
from PIL import Image
import yolo_utils # Imports the utility functions from your project

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YOLOv3 Object Detection with Streamlit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODEL LOADING ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_yolo_model():
    """
    Loads the YOLOv3 model and class names from the project files.
    """
    try:
        # Define paths based on your project structure
        config_path = "yolov3-coco/yolov3.cfg"
        weights_path = "yolov3-coco/yolov3.weights"
        labels_path = "yolov3-coco/coco-labels"
        
        # Load the class names
        labels = open(labels_path).read().strip().split('\n')
        
        # Load the YOLOv3 network
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        
        return net, labels
    except Exception as e:
        st.error(f"Error loading the YOLOv3 model: {e}")
        st.error("Please ensure the following files exist in the 'yolov3-coco' folder: yolov3.cfg, yolov3.weights, coco-labels")
        return None, None

# Load the model and labels
net, LABELS = load_yolo_model()

# --- UI COMPONENTS ---
st.title("🤖 YOLOv3 Object Detection")
st.write(
    "Upload an image and the YOLOv3 model will detect objects in it. "
    "This app uses the OpenCV DNN module."
)

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and net is not None:
    # Read the uploaded image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)

    # Convert the image from BGR (OpenCV default) to RGB for display
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Perform object detection using your project's utility functions
    (H, W) = original_image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(original_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    # Process detections
    boxes, confidences, class_ids = yolo_utils.extract_boxes_confidences_classids(layer_outputs, 0.5, W, H)
    idxs = yolo_utils.suppress_weak_overlapping_boxes(boxes, confidences, 0.5, 0.3)
    
    # Draw the bounding boxes on the image
    if len(idxs) > 0:
        yolo_utils.draw_bounding_boxes(original_image, LABELS, idxs, boxes, confidences, class_ids)
    
    rgb_combined_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # --- DISPLAY RESULTS ---
    st.header("Detection Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(rgb_image, caption="The image you uploaded.", use_column_width=True)

    with col2:
        st.subheader("Image with Detected Objects")
        st.image(rgb_combined_image, caption="Objects detected by YOLOv3.", use_column_width=True)

    # Display detected object classes
    if len(idxs) > 0:
        detected_classes = [LABELS[class_ids[i]] for i in idxs.flatten()]
        st.write("---")
        st.subheader("Detected Objects:")
        st.write(f"Found **{len(detected_classes)}** objects.")
        st.write(", ".join(set(detected_classes))) # Show unique class names
    else:
        st.write("---")
        st.info("No objects were detected in the image.")

elif net is None:
    st.warning("Model could not be loaded. Please check the error message above.")

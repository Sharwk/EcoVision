# Filename: app.py
# Author: Sean Bell
# Created: 21 October 2025
# School: Colby College
# For CS366 Final Project: EcoVision
# Professor Chowdhury

# Description: Streamlit interface for EcoVision trash detection


import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import supervision as sv
import cv2
import tempfile
import os
from pathlib import Path
import numpy as np
from io import BytesIO

# Initialize annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Setting page layout
st.set_page_config(
    page_title="EcoVision - Trash Detection",
    page_icon="icons/ecoeye.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Main Title
st.title("EcoVision - Trash Detection")
st.sidebar.image("icons/ecovision2.png", width=200)
st.sidebar.header("Model Configuration")

# Model configuration
confidence = st.sidebar.slider(
    "Select Confidence Level", 
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)

#Slider note, font made smaller and pushed closer to slider
st.sidebar.markdown(
    """
    <p style="
        font-size: 0.75rem;
        opacity: 0.85;
        margin-top: -5px;   
        margin-bottom: 30px; /* adds space before next  */
    ">
        Controls how sure EcoVision must be before it shows a detection.
    </p>
    """,
    unsafe_allow_html=True
)




# Source type selection
source_type = st.sidebar.radio("Select Source Type", ["Image", "Video"])
model_choice = st.sidebar.radio("Select Model",["EcoVision Core", "EcoVision Pro"])

if model_choice == "EcoVision Pro":
    st.sidebar.warning("EcoVision Pro will be avalaible for purchase soon! 🚀")
    st.session_state.model_choice = "EcoVision Core" # force fallback lol
    st.rerun()

# Load YOLO model
@st.cache_resource
def main_model():
    model = YOLO('models/best.pt')
    return model

def process_uploaded_video(video_bytes):
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(video_bytes)
        video_path = tmpfile.name

    try:
        # Read the video
        cap = cv2.VideoCapture(video_path)        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_tmpfile:
            output_path = output_tmpfile.name
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load model
        model = main_model()

        st.success("Processing video")        
        # Process frames
        progress_bar = st.progress(0)
        frame_count = 0
        total_class_counts = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = model(frame, conf=confidence)
            processed_frame = results[0].plot()
            
            # Update class counts
            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                total_class_counts[class_name] = total_class_counts.get(class_name, 0) + 1
            
            # Write processed frame
            out.write(processed_frame)
            
            # Update progress
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        # Release resources
        cap.release()
        out.release()
        
        # Read the processed video into memory
        with open(output_path, 'rb') as f:
            processed_video_bytes = f.read()
            
        return processed_video_bytes, total_class_counts
        
    finally:
        # Clean up temporary files
        if os.path.exists(video_path):
            os.unlink(video_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

if source_type == "Image":
    # Image processing code
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', "bmp", "webp"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image",
                    use_container_width=True)
        else:
            st.markdown(
                '<div class="upload-info">Please upload an image.</div>',
                unsafe_allow_html=True)
            

            
    with col2:
        if st.sidebar.button('Detect') and uploaded_file is not None:
            try:
                model = main_model()
                results = model(
                    source=uploaded_image,
                    conf=confidence,
                    device="cpu"
                )
                boxes = results[0].boxes
                res_plotted = results[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detect Image',
                         use_container_width=True)
                                
                # Count objects
                class_counts = {}
                for cls in boxes.cls:
                    class_name = model.names[int(cls)]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                
                # Display counts
                table_data = [{"Class": class_name, "Count": count} 
                             for class_name, count in class_counts.items()]
                st.write("Number of objects for each detected class:")
                st.table(table_data)
                
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.exception(ex)
        else:
            st.markdown(
                '<div class="upload-error">Please upload an image first!</div>',
                unsafe_allow_html=True
            )

else:  # Video processing
    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Display original video in a smaller size
        video_bytes = uploaded_file.read()
        
        # Create a container with custom width
        video_container = st.container()
        with video_container:
            col1, col2, col3 = st.columns([1,2,1])  # Creates three columns with middle one being larger
            with col2:  # Use the middle column for the video
                st.video(video_bytes)
        
        if st.sidebar.button('Detect'):
            try:
                # Process video
                processed_video_bytes, total_class_counts = process_uploaded_video(video_bytes)
                
                # Display detection statistics
                st.write("Total detections throughout the video:")
                table_data = [{"Class": class_name, "Total Count": count} 
                             for class_name, count in total_class_counts.items()]
                st.table(table_data)
                
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.download_button(
                        label="⬇️ Download processed video",
                        data=processed_video_bytes,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )
                
            except Exception as ex:
                st.exception(ex)
                st.error("An error occurred during video processing. Please try again.")
    else:
        st.markdown(
            '<div class="upload-error">Please upload a video first!</div>',
            unsafe_allow_html=True
        )

st.sidebar.markdown(
    """
    <div style="
        position: fixed;
        bottom: 15px;
        left: 15px;
        color: white;
        font-size: 0.85rem;
        opacity: 0.7;
        z-index: 999;
    ">
        By Sean Bell
    </div>
    """,
    unsafe_allow_html=True
)




# ---- EcoVision theme ----
st.markdown(
    """
    <style>
    /* MAIN & SIDEBAR BACKGROUNDS + TEXT */

    .stApp {
        background-color: #3F4F44;      /* main background: normal green */
        color: #ffffff;                 /* default text white */
    }

    section[data-testid="stSidebar"] {
        background-color: #2C3930;      /* sidebar: dark green */
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;      /* sidebar text white */
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;                 /* all headings white */
    }

    .stMarkdown p {
        color: #ffffff;                 /* body text white */
    }

    /* FILE UPLOADER (drop area + button) */

    div[data-testid="stFileUploaderDropzone"] {
        background-color: #15803d;      /* slightly darker green block */
        border: 2px dashed #ffffff;     /* white dashed border */
        border-radius: 0.75rem;
    }

    div[data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;      /* text inside dropzone white */
    }

    /* "Browse files" button inside uploader */
    div[data-testid="stFileUploader"] button {
        background-color: #A27B5C;      /* light blue */
        color: #FFFFFF;                 /* dark text */
        border-radius: 999px;
        border: none;
        font-weight: 600;
        padding: 0.25rem 1.1rem;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #c4fa9c;      /* brown */
        color: #0f172a;
    }

/* TOP HEADER BAR (the black bar with the menu + deploy buttons) */
header[data-testid="stHeader"] {
    background-color: #A27B5C !important;  
    color: #ffffff !important;
    border-bottom: 1px solid rgba(255,255,255,0.15);
}

/* Make header text/buttons white */
header[data-testid="stHeader"] * {
    color: #ffffff !important;
}

    /* DETECT BUTTON (sidebar) */

    .stButton > button {
        background-color: #A27B5C;      /* light blue */
        color: #FFFFFF;
        border-radius: 999px;
        border: none;
        font-weight: 600;
        padding: 0.25rem 1.2rem;
    }
    .stButton > button:hover {
        background-color: #c4fa9c;
        color: #0f172a;
    }

/* FIX STREAMLIT SLIDER COLORS COMPLETELY */

/* Entire slider container */
div[data-baseweb="slider"] {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}

/* Inactive track (the gray bar) */
div[data-baseweb="slider"] > div > div {
    background: #475569 !important;            /* Slate gray */
    height: 6px !important;
    border-radius: 4px !important;
}

/* Active (filled) track */
div[data-baseweb="slider"] [data-baseweb="track"] > div {
    background: #c4fa9c !important;            /* LIGHT BLUE YOU WANT */
    height: 6px !important;
    border-radius: 4px !important;
}

/* Slider thumb / handle */
div[data-baseweb="slider"] [role="slider"] {
    background: #A27B5C !important;            /* Green */
    border: 3px solid #FFFFFF !important;      /* WHITE border */
    width: 18px !important;
    height: 18px !important;
    box-shadow: none !important;
    border-radius: 50% !important;
}

/* Fix hover state */
div[data-baseweb="slider"] [role="slider"]:hover {
    background: #c4fa9c !important;            /* brighter blue */
}

/* Prevent Streamlit from painting red when dragging */
div[data-baseweb="slider"] [data-baseweb="thumb"] {
    background: #0ea5e9 !important;
}


/* label text */
div[role="radiogroup"] > label > p {
    color: #ffffff !important;
}

    /* CUSTOM "PLEASE UPLOAD..." BOXES */

    .upload-info {
        background-color: #DCD7C9;      /* light blue */
        color: #0f172a;
        padding: 0.6rem 1rem;
        border-radius: 0.5rem;
        font-weight: 500;
        text-align: center;
    }

    .upload-error {
        background-color: #475569;      /* gray */
        color: #f9fafb;
        padding: 0.6rem 1rem;
        border-radius: 0.5rem;
        font-weight: 500;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

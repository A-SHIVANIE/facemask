import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('mask_detector.keras')
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels_dict = {0: 'MASK', 1: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

st.title("Face Mask Detection App")

# Select mode: real-time webcam or file upload
mode = st.radio("Select mode:", ("Real-time Webcam", "Upload Image/Video"))

if mode == "Real-time Webcam":
    run = st.checkbox('Run the webcam')
    
    if run:
        st.write("Starting Webcam...")
        cap = cv2.VideoCapture(0)  # Open the default camera
        frame_window = st.empty()  # Create a placeholder for the video frame

        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Unable to capture video.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                resized = cv2.resize(face_img, (100, 100))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 100, 100, 1))
                
                # Ensure proper prediction call
                with st.spinner("Detecting mask..."):
                    result = model.predict(reshaped)

                label = np.argmax(result, axis=1)[0]
                color = color_dict[label]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Display the video frame in Streamlit
            frame_window.image(frame, channels="BGR")

        cap.release()

elif mode == "Upload Image/Video":
    # Drag and drop file uploader
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            # Process uploaded image
            img = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y + h, x:x + w]
                resized = cv2.resize(face_img, (100, 100))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 100, 100, 1))
                
                # Ensure proper prediction call
                with st.spinner("Detecting mask..."):
                    result = model.predict(reshaped)

                label = np.argmax(result, axis=1)[0]
                color = color_dict[label]

                cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_rgb, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            st.image(img_rgb, channels="RGB")

        elif uploaded_file.type == "video/mp4":
            # Process uploaded video
            st.write("Processing video...")
            cap = cv2.VideoCapture(uploaded_file.name)

            frame_window = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_img = gray[y:y + h, x:x + w]
                    resized = cv2.resize(face_img, (100, 100))
                    normalized = resized / 255.0
                    reshaped = np.reshape(normalized, (1, 100, 100, 1))
                    
                    # Ensure proper prediction call
                    with st.spinner("Detecting mask..."):
                        result = model.predict(reshaped)

                    label = np.argmax(result, axis=1)[0]
                    color = color_dict[label]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                frame_window.image(frame, channels="BGR")

            cap.release()

# Add stop button if required
if st.button("Stop", key="stop_button"):
    st.write("Stopping...")
























from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import json
from ultralytics import YOLO
import google.generativeai as genai
from gtts import gTTS
import io
import asyncio
from fastapi.responses import StreamingResponse

# Configure Gemini API
api_key = "AIzaSyBSCp3SG9pBiAFvo9e5zVupU4D4Nhoyd-o"
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 20,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

# Load Gemini model
model_gemini = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def generate_content(prompt):
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Error with Gemini API:", str(e))
        return "I encountered an error while processing your request."

# Load YOLO model
model_yolo = YOLO('yolov10n.pt')

def process_frame(frame):
    results = model_yolo(frame)
    detected_objects = []

    for box in results[0].boxes:
        xyxy = box.xyxy.cpu().numpy()
        confidence = float(box.conf.cpu().numpy().item())
        class_id = int(box.cls.cpu().numpy().item())
        label = model_yolo.names[class_id]

        x_min, y_min, x_max, y_max = map(int, xyxy.flatten())
        object_center_x = (x_min + x_max) // 2
        object_center_y = (y_min + y_max) // 2
        horizontal_position = "left" if object_center_x < frame.shape[1] // 2 else "right"
        vertical_position = "up" if object_center_y < frame.shape[0] // 2 else "down"
        position_description = f"{label} ({confidence:.2f}) is {horizontal_position} and {vertical_position}"
        detected_objects.append(position_description)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame, detected_objects

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    # Use write_to_fp to write the audio to the BytesIO object
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)  # Rewind the file pointer to the start
    return audio_io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            image_data = data  # Get the image data (base64 string)

            # Convert the base64 string to an image
            image_data = data.split(",")[1]  # If the image data has a prefix, like "data:image/png;base64,"
            img_data = base64.b64decode(image_data)
            np_array = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            # Process the frame using YOLO model
            frame, detected_objects = process_frame(frame)

            # Generate content using Gemini based on detected objects
            if detected_objects:
                # Create a natural language description with a pre-prompt
                pre_prompt = "You are an AI assistant for a blind person. Describe the surroundings based on detected objects in a helpful and concise manner with a assumed distance and direction of objects."
                object_description = ". ".join(detected_objects)
                prompt = f"{pre_prompt} The detected objects are: {object_description}. Please describe them."
                print("\nPrompt to Gemini API:", prompt)

                # Get the response from Gemini
                content = generate_content(prompt)
                print("\nGemini's Response:\n", content)

                # Convert the response to speech
                audio_io = text_to_speech(content)

                # Send the response as a JSON message
                response = {
                    "objects": detected_objects,
                    "geminiResponse": content
                }

                # Send the response and the audio back to the client
                await websocket.send_text(json.dumps(response))

                # Sending audio as a streaming response in WebSocket
                await websocket.send_bytes(audio_io.read())

            else:
                response = {"objects": [], "geminiResponse": "No objects detected."}
                await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Client disconnected")

# Add a basic route for testing the API
@app.get("/")
def read_root():
    return {"message": "Welcome to the Anshu Gemini API"}
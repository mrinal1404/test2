import cv2
import mediapipe as mp
import numpy as np
import socket
import threading
import base64
import json
import requests
import tkinter as tk
from tkinter import messagebox, simpledialog

class GestureVideoCallApp:
    def __init__(self, host='0.0.0.0', port=65432):
        # UI Setup
        self.root = None
        
        # Video and Gesture Recognition Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Network Setup
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Gesture Translation Dictionary (Expanded)
        self.gesture_translations = {
            'peace': '✌️ Peace / Hello',
            'thumbs_up': '👍 Thumbs Up / Great!',
            'open_palm': '✋ Open Palm / Stop / Wait',
            'fist': '✊ Closed Fist / OK / Understood',
            'point': '👉 Pointing / Look Here',
            'wave': '👋 Waving / Hi'
        }

    def create_connection_ui(self):
        """Create a GUI for connection setup"""
        self.root = tk.Tk()
        self.root.title("Gesture Video Call")
        self.root.geometry("300x250")

        # Connection Mode Selection
        tk.Label(self.root, text="Select Connection Mode", font=("Arial", 12)).pack(pady=10)
        
        server_btn = tk.Button(self.root, text="Start Server", command=self.start_server_mode)
        server_btn.pack(pady=5)
        
        client_btn = tk.Button(self.root, text="Join Call", command=self.start_client_mode)
        client_btn.pack(pady=5)

        self.root.mainloop()

    def start_server_mode(self):
        """Start server mode with IP display"""
        if self.root:
            self.root.destroy()
        
        # Get external IP
        try:
            external_ip = requests.get('https://api.ipify.org').text
        except:
            external_ip = socket.gethostbyname(socket.gethostname())

        # Create server connection window
        tk.messagebox.showinfo("Server Info", 
            f"Server Details:\n"
            f"Local IP: {socket.gethostbyname(socket.gethostname())}\n"
            f"External IP: {external_ip}\n"
            f"Port: {self.port}\n"
            "Share these details with your call partner!")

        # Start server threading
        server_thread = threading.Thread(target=self._start_server)
        server_thread.start()

    def _start_server(self):
        """Internal server startup method"""
        self.socket.bind((self.host, self.port))
        self.socket.listen(2)
        print("Server waiting for connections...")

        # Accept two clients
        clients = []
        while len(clients) < 2:
            client, address = self.socket.accept()
            clients.append(client)
            print(f"Connected to {address}")

        # Start video streaming
        self._handle_video_stream(clients[0], clients[1])

    def start_client_mode(self):
        """Start client connection mode"""
        if self.root:
            self.root.destroy()
        
        # Prompt for server IP
        server_ip = simpledialog.askstring("Connection", "Enter Server IP Address:")
        
        if not server_ip:
            messagebox.showerror("Error", "Invalid IP Address")
            return

        try:
            self.socket.connect((server_ip, self.port))
            print(f"Connected to server at {server_ip}")
            
            # Start client video streaming
            self._start_client_stream()
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def _start_client_stream(self):
        """Client-side video streaming method"""
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect gesture
            gesture, annotated_frame = self.detect_gesture(frame)

            # Encode frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = base64.b64encode(buffer)

            # Prepare payload
            payload = {
                'frame': frame_bytes.decode('utf-8'),
                'gesture': self.gesture_translations.get(gesture, '')
            }

            # Send to server
            try:
                self.socket.send(json.dumps(payload).encode('utf-8'))
            except:
                break

            # Receive and display incoming frame
            try:
                data = self.socket.recv(1024 * 1024)
                incoming_payload = json.loads(data.decode('utf-8'))
                
                # Decode incoming frame
                incoming_frame = base64.b64decode(incoming_payload['frame'])
                frame_array = np.frombuffer(incoming_frame, dtype=np.uint8)
                img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # Display gesture translation
                if incoming_payload['gesture']:
                    cv2.putText(
                        img, 
                        incoming_payload['gesture'], 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                
                cv2.imshow('Incoming Video', img)
            except:
                pass

            # Display local video
            cv2.imshow('Local Video', annotated_frame)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.socket.close()

    def detect_gesture(self, frame):
        """Detect hand gestures in the video frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = self._classify_gesture(hand_landmarks)
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return gesture, frame

    def _classify_gesture(self, hand_landmarks):
        """Advanced gesture classification"""
        landmarks = hand_landmarks.landmark
        
        # Expanded gesture recognition
        if self._is_peace_sign(landmarks):
            return 'peace'
        elif self._is_thumbs_up(landmarks):
            return 'thumbs_up'
        elif self._is_open_palm(landmarks):
            return 'open_palm'
        elif self._is_closed_fist(landmarks):
            return 'fist'
        elif self._is_pointing(landmarks):
            return 'point'
        elif self._is_waving(landmarks):
            return 'wave'
        
        return None

    def _is_peace_sign(self, landmarks):
        """Detect peace sign gesture"""
        return (landmarks[8].y < landmarks[6].y and 
                landmarks[12].y < landmarks[10].y and
                landmarks[4].x > landmarks[3].x)

    def _is_thumbs_up(self, landmarks):
        """Detect thumbs up gesture"""
        return (landmarks[4].y < landmarks[3].y and 
                landmarks[8].y > landmarks[7].y)

    def _is_open_palm(self, landmarks):
        """Detect open palm gesture"""
        return all(landmarks[i].y < landmarks[i-1].y 
                   for i in [8, 12, 16, 20])

    def _is_closed_fist(self, landmarks):
        """Detect closed fist gesture"""
        return all(landmarks[i].y > landmarks[i-1].y 
                   for i in [8, 12, 16, 20])

    def _is_pointing(self, landmarks):
        """Detect pointing gesture"""
        return (landmarks[8].y < landmarks[7].y and 
                all(landmarks[i].y > landmarks[i-1].y 
                    for i in [12, 16, 20]))

    def _is_waving(self, landmarks):
        """Detect waving gesture"""
        return (landmarks[8].x > landmarks[6].x and 
                landmarks[4].x < landmarks[2].x)

def main():
    """Main application entry point"""
    app = GestureVideoCallApp()
    app.create_connection_ui()

if __name__ == "__main__":
    main()
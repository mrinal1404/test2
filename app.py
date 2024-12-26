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
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture Video Call")
        self.root.geometry("300x300")
        
        self.host = '0.0.0.0'
        self.port = 65432
        self.connection_timeout = 10
        self.is_connected = False
        self.should_run = True
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.gesture_translations = {
            'peace': '✌️ Peace / Hello',
            'thumbs_up': '👍 Thumbs Up / Great!',
            'open_palm': '✋ Open Palm / Stop / Wait',
            'fist': '✊ Closed Fist / OK / Understood',
            'point': '👉 Pointing / Look Here',
            'wave': '👋 Waving / Hi'
        }
        
        self.setup_ui()
        self.receive_buffer = bytearray()

    def setup_ui(self):
        self.status_label = tk.Label(self.root, text="Ready to connect", font=("Arial", 10))
        self.status_label.pack(pady=5)

        tk.Label(self.root, text="Select Connection Mode", font=("Arial", 12)).pack(pady=10)
        
        server_btn = tk.Button(self.root, text="Start Server", command=self.start_server_mode)
        server_btn.pack(pady=5)
        
        client_btn = tk.Button(self.root, text="Join Call", command=self.start_client_mode)
        client_btn.pack(pady=5)

        exit_btn = tk.Button(self.root, text="Exit", command=self.cleanup_and_exit)
        exit_btn.pack(pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self.cleanup_and_exit)

    def update_status(self, message):
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)

    def create_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(self.connection_timeout)

    def start_server_mode(self):
        self.create_socket()
        self.update_status("Starting server...")
        
        try:
            external_ip = requests.get('https://api.ipify.org', timeout=5).text
        except:
            external_ip = "Could not determine external IP"
        
        local_ip = socket.gethostbyname(socket.gethostname())
        
        self.root.withdraw()
        
        messagebox.showinfo("Server Info", 
            f"Server Details:\n"
            f"Local IP: {local_ip}\n"
            f"External IP: {external_ip}\n"
            f"Port: {self.port}")
        
        server_thread = threading.Thread(target=self._start_server)
        server_thread.daemon = True
        server_thread.start()

    def _start_server(self):
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(2)
            self.update_status("Waiting for connections...")
            
            client, address = self.socket.accept()
            self.is_connected = True
            self._handle_video_stream(client)
            
        except Exception as e:
            self.update_status(f"Server error: {str(e)}")
            messagebox.showerror("Server Error", str(e))

    def start_client_mode(self):
        self.root.withdraw()
        server_ip = simpledialog.askstring("Connection", "Enter Server IP Address:")
        
        if not server_ip:
            self.root.deiconify()
            return
            
        try:
            self.create_socket()
            self.update_status("Connecting...")
            self.socket.connect((server_ip, self.port))
            self.is_connected = True
            self._start_client_stream()
        except Exception as e:
            self.update_status(f"Connection error: {str(e)}")
            messagebox.showerror("Connection Error", str(e))
            self.root.deiconify()

    def _handle_video_stream(self, client):
        cap = cv2.VideoCapture(0)
        receive_thread = threading.Thread(target=self._receive_stream, args=(client,))
        receive_thread.daemon = True
        receive_thread.start()
        
        while self.should_run and self.is_connected:
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                gesture, annotated_frame = self.detect_gesture(frame)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = base64.b64encode(buffer)

                payload = {
                    'frame': frame_bytes.decode('utf-8'),
                    'gesture': self.gesture_translations.get(gesture, '')
                }
                
                client.send(json.dumps(payload).encode('utf-8'))
                
                cv2.imshow('Local Video', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Stream error: {e}")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.cleanup()

    def _start_client_stream(self):
        cap = cv2.VideoCapture(0)
        receive_thread = threading.Thread(target=self._receive_stream, args=(self.socket,))
        receive_thread.daemon = True
        receive_thread.start()
        
        while self.should_run and self.is_connected:
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                gesture, annotated_frame = self.detect_gesture(frame)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = base64.b64encode(buffer)

                payload = {
                    'frame': frame_bytes.decode('utf-8'),
                    'gesture': self.gesture_translations.get(gesture, '')
                }
                
                self.socket.send(json.dumps(payload).encode('utf-8'))
                
                cv2.imshow('Local Video', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Stream error: {e}")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.cleanup()

    def _receive_stream(self, connection):
        while self.should_run and self.is_connected:
            try:
                data = connection.recv(65536)
                if not data:
                    break
                
                self.receive_buffer.extend(data)
                
                while True:
                    try:
                        payload = json.loads(self.receive_buffer.decode())
                        frame_data = base64.b64decode(payload['frame'])
                        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        
                        cv2.imshow('Remote Video', frame)
                        cv2.waitKey(1)
                        
                        if payload['gesture']:
                            print(f"Remote gesture: {payload['gesture']}")
                            
                        self.receive_buffer = bytearray()
                        break
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        break
                    
            except Exception as e:
                print(f"Receive error: {e}")
                break

    def detect_gesture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = self._classify_gesture(hand_landmarks)
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return gesture, frame

    def _classify_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
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
        return (landmarks[8].y < landmarks[6].y and 
                landmarks[12].y < landmarks[10].y and
                landmarks[4].x > landmarks[3].x)

    def _is_thumbs_up(self, landmarks):
        return (landmarks[4].y < landmarks[3].y and 
                landmarks[8].y > landmarks[7].y)

    def _is_open_palm(self, landmarks):
        return all(landmarks[i].y < landmarks[i-1].y 
                   for i in [8, 12, 16, 20])

    def _is_closed_fist(self, landmarks):
        return all(landmarks[i].y > landmarks[i-1].y 
                   for i in [8, 12, 16, 20])

    def _is_pointing(self, landmarks):
        return (landmarks[8].y < landmarks[7].y and 
                all(landmarks[i].y > landmarks[i-1].y 
                    for i in [12, 16, 20]))

    def _is_waving(self, landmarks):
        return (landmarks[8].x > landmarks[6].x and 
                landmarks[4].x < landmarks[2].x)

    def cleanup(self):
        self.should_run = False
        self.is_connected = False
        
        if hasattr(self, 'socket'):
            try:
                self.socket.close()
            except:
                pass
            
        cv2.destroyAllWindows()

    def cleanup_and_exit(self):
        self.cleanup()
        if self.root:
            self.root.quit()
            self.root.destroy()

    def run(self):
        self.root.mainloop()

def main():
    app = GestureVideoCallApp()
    app.run()

if __name__ == "__main__":
    main()
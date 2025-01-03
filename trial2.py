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
from typing import Optional, Tuple, Dict, Any
import time

class GestureVideoCallApp:
    def __init__(self, host: str = '0.0.0.0', port: int = 65432):
        # Core settings
        self.host = host
        self.port = port
        self.connection_timeout = 30
        self.is_connected = False
        self.should_run = True
        self.receive_buffer = bytearray()

        # Initialize UI
        self.root = tk.Tk()
        self.root.title("Gesture Video Call")
        self.root.geometry("400x400")
        self.setup_ui()

        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Gesture definitions with confidence thresholds
        self.gesture_translations = {
            'peace': '✌️ Peace / Hello',
            'thumbs_up': '👍 Thumbs Up / Great!',
            'open_palm': '✋ Open Palm / Stop / Wait',
            'fist': '✊ Closed Fist / OK / Understood',
            'point': '👉 Pointing / Look Here',
            'wave': '👋 Waving / Hi'
        }

    def setup_ui(self):
        """Initialize enhanced UI components with better styling and feedback"""
        # Status display
        self.status_frame = tk.Frame(self.root, relief=tk.SUNKEN, borderwidth=1)
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = tk.Label(
            self.status_frame, 
            text="Ready to connect", 
            font=("Arial", 10),
            pady=5
        )
        self.status_label.pack()

        # Connection controls
        tk.Label(
            self.root,
            text="Gesture Video Call",
            font=("Arial", 16, "bold")
        ).pack(pady=10)

        # Connection mode frame
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=10)

        server_btn = tk.Button(
            mode_frame,
            text="Start Server",
            command=self.start_server_mode,
            width=15,
            height=2
        )
        server_btn.pack(side=tk.LEFT, padx=5)

        client_btn = tk.Button(
            mode_frame,
            text="Join Call",
            command=self.start_client_mode,
            width=15,
            height=2
        )
        client_btn.pack(side=tk.LEFT, padx=5)

        # Settings and controls
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(pady=10)

        self.quality_var = tk.StringVar(value="medium")
        quality_label = tk.Label(controls_frame, text="Video Quality:")
        quality_label.pack()
        
        qualities = ["low", "medium", "high"]
        for q in qualities:
            tk.Radiobutton(
                controls_frame,
                text=q.capitalize(),
                variable=self.quality_var,
                value=q
            ).pack()

        # Exit button
        exit_btn = tk.Button(
            self.root,
            text="Exit",
            command=self.cleanup_and_exit,
            width=15,
            bg="red",
            fg="white"
        )
        exit_btn.pack(pady=20)

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup_and_exit)

    def update_status(self, message: str, level: str = "info"):
        """Update status with color-coding based on message type"""
        colors = {
            "info": "black",
            "success": "green",
            "error": "red",
            "warning": "orange"
        }
        if hasattr(self, 'status_label'):
            self.status_label.config(
                text=message,
                fg=colors.get(level, "black")
            )

    def create_socket(self):
        """Create and configure socket with error handling"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(self.connection_timeout)
        except socket.error as e:
            self.update_status(f"Socket creation failed: {e}", "error")
            raise

    def start_server_mode(self):
        """Enhanced server mode with better error handling and feedback"""
        try:
            self.create_socket()
            self.update_status("Starting server...", "info")
            
            # Get IP addresses
            external_ip = requests.get('https://api.ipify.org', timeout=5).text
            local_ip = socket.gethostbyname(socket.gethostname())
            
            # Show connection info
            server_info = (
                f"Server Details:\n\n"
                f"Local IP: {local_ip}\n"
                f"External IP: {external_ip}\n"
                f"Port: {self.port}\n\n"
                f"Share these details with the person you want to call."
            )
            
            messagebox.showinfo("Server Information", server_info)
            
            # Start server thread
            server_thread = threading.Thread(target=self._start_server)
            server_thread.daemon = True
            server_thread.start()
            
        except Exception as e:
            self.update_status(f"Server startup failed: {str(e)}", "error")
            messagebox.showerror("Error", f"Failed to start server: {str(e)}")

    def _start_server(self):
        """Internal server handler with improved connection management"""
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(2)
            self.update_status("Waiting for incoming connection...", "info")
            
            while self.should_run:
                client, address = self.socket.accept()
                self.update_status(f"Connected to {address[0]}", "success")
                self.is_connected = True
                self._handle_video_stream(client)
                
        except Exception as e:
            self.update_status(f"Server error: {str(e)}", "error")
        finally:
            self.cleanup()

    def start_client_mode(self):
        """Enhanced client mode with connection validation"""
        self.root.withdraw()
        server_ip = simpledialog.askstring(
            "Connect to Call",
            "Enter the server's IP address:"
        )
        
        if not server_ip:
            self.root.deiconify()
            return
            
        try:
            self.create_socket()
            self.update_status("Connecting to server...", "info")
            self.socket.connect((server_ip, self.port))
            self.is_connected = True
            self.update_status("Connected!", "success")
            self._start_client_stream()
            
        except socket.timeout:
            self.update_status("Connection timed out", "error")
            messagebox.showerror("Error", "Connection timed out. Please verify the IP and try again.")
        except ConnectionRefusedError:
            self.update_status("Connection refused", "error")
            messagebox.showerror("Error", "Connection refused. Please verify the server is running.")
        except Exception as e:
            self.update_status(f"Connection failed: {str(e)}", "error")
            messagebox.showerror("Error", f"Failed to connect: {str(e)}")
        finally:
            self.root.deiconify()

    def _start_client_stream(self):
        """Handle client-side video streaming"""
        try:
            cap = cv2.VideoCapture(0)
            
            # Set video quality based on user selection
            quality_settings = {
                "low": (640, 480, 15),
                "medium": (1280, 720, 30),
                "high": (1920, 1080, 30)
            }
            
            width, height, fps = quality_settings[self.quality_var.get()]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            # Start receive thread
            receive_thread = threading.Thread(target=self._receive_stream, args=(self.socket,))
            receive_thread.daemon = True
            receive_thread.start()
            
            while self.should_run and self.is_connected:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gesture, annotated_frame = self.detect_gesture(frame)
                _, buffer = cv2.imencode('.jpg', annotated_frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = base64.b64encode(buffer)
                
                payload = {
                    'frame': frame_bytes.decode('utf-8'),
                    'gesture': self.gesture_translations.get(gesture, ''),
                    'timestamp': time.time()
                }
                
                self.socket.send(json.dumps(payload).encode('utf-8'))
                
                cv2.imshow('Local Video', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            self.update_status(f"Stream error: {str(e)}", "error")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()

    def _receive_stream(self, connection):
        """Handle receiving video stream data"""
        while self.should_run and self.is_connected:
            try:
                data = connection.recv(65536)
                if not data:
                    break
                
                self.receive_buffer.extend(data)
                
                try:
                    payload = json.loads(self.receive_buffer.decode())
                    frame_data = base64.b64decode(payload['frame'])
                    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Display received gesture if present
                        if payload.get('gesture'):
                            cv2.putText(
                                frame,
                                payload['gesture'],
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2
                            )
                        
                        cv2.imshow('Remote Video', frame)
                        cv2.waitKey(1)
                    
                    self.receive_buffer = bytearray()
                    
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Incomplete data, continue receiving
                    continue
                    
            except Exception as e:
                print(f"Receive error: {e}")
                break
        
        self.cleanup()

    def _handle_video_stream(self, client: socket.socket):
        """Enhanced video stream handler with quality settings"""
        cap = cv2.VideoCapture(0)
        
        # Set video quality based on user selection
        quality_settings = {
            "low": (640, 480, 15),
            "medium": (1280, 720, 30),
            "high": (1920, 1080, 30)
        }
        
        width, height, fps = quality_settings[self.quality_var.get()]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Start receive thread
        receive_thread = threading.Thread(target=self._receive_stream, args=(client,))
        receive_thread.daemon = True
        receive_thread.start()
        
        while self.should_run and self.is_connected:
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame and detect gestures
                gesture, annotated_frame = self.detect_gesture(frame)
                _, buffer = cv2.imencode('.jpg', annotated_frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = base64.b64encode(buffer)

                # Prepare and send payload
                payload = {
                    'frame': frame_bytes.decode('utf-8'),
                    'gesture': self.gesture_translations.get(gesture, ''),
                    'timestamp': time.time()
                }
                
                client.send(json.dumps(payload).encode('utf-8'))
                
                # Display local video
                cv2.imshow('Local Video', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                self.update_status(f"Stream error: {str(e)}", "error")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.cleanup()

    def detect_gesture(self, frame: np.ndarray) -> Tuple[Optional[str], np.ndarray]:
        """Enhanced gesture detection with improved accuracy"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks with enhanced styling
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Classify gesture
                gesture = self._classify_gesture(hand_landmarks)
                
                # Add gesture label to frame
                if gesture:
                    cv2.putText(
                        frame,
                        self.gesture_translations.get(gesture, ''),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
        
        return gesture, frame

    def _classify_gesture(self, landmarks) -> Optional[str]:
        """Enhanced gesture classification with improved accuracy"""
        # Convert landmarks to numpy array for easier calculations
        points = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
        
        # Calculate key angles and distances for better gesture recognition
        if self._is_peace_sign(points):
            return 
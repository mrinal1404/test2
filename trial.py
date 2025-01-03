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
            return 'peace'
        elif self._is_thumbs_up(points):
            return 'thumbs_up'
        elif self._is_open_palm(points):
            return 'open_palm'
        elif self._is_closed_fist(points):
            return 'fist'
        elif self._is_pointing(points):
            return 'point'
        elif self._is_waving(points):
            return 'wave'
        
        return None

    # Improved gesture detection methods with better accuracy
    def _is_peace_sign(self, points: np.ndarray) -> bool:
        return (points[8, 1] < points[6, 1] and  # Index finger up
                points[12, 1] < points[10, 1] and  # Middle finger up
                points[16, 1] > points[14, 1] and  # Ring finger down
                points[20, 1] > points[18, 1])     # Pinky down

    def _is_thumbs_up(self, points: np.ndarray) -> bool:
        return (points[4, 1] < points[3, 1] and    # Thumb up
                all(points[i, 1] > points[i-1, 1]   # Other fingers down
                    for i in [8, 12, 16, 20]))

    def _is_open_palm(self, points: np.ndarray) -> bool:
        return (all(points[i, 1] < points[i-1, 1]   # All fingers up
                   for i in [8, 12, 16, 20]) and
                points[4, 0] < points[3, 0])        # Thumb position

    def _is_closed_fist(self, points: np.ndarray) -> bool:
        return (all(points[i, 1] > points[i-2, 1]   # All fingers curled
                   for i in [8, 12, 16, 20]) and
                points[4, 0] > points[3, 0])        # Thumb position

    def _is_pointing(self, points: np.ndarray) -> bool:
        return (points[8, 1] < points[6, 1] and    # Index finger up
                all(points[i, 1] > points[i-1, 1]   # Other fingers down
                    for i in [12, 16, 20]))

    def _is_waving(self, points: np.ndarray) -> bool:
        return (points[8, 0] > points[6, 0] and    # Fingers spread
                points[4, 0] < points[2, 0] and    # Thumb position
                all(points[i, 1] < points[i-1, 1]   # Fingers up
                    for i in [8, 12, 16, 20]))

    def cleanup(self):
        """Enhanced cleanup with proper resource management"""
        self.should_run = False
        self.is_connected = False
        
        if hasattr(self, 'socket'):
            try:
                self.socket.shutdown(socket.SHUT_RDWR)
                self.socket.close()
            except:
                pass
            
        cv2.destroyAllWindows()
        self.update_status("Disconnected", "warning")

    def cleanup_and_exit(self):
        """Clean exit with proper cleanup"""
        self.cleanup()
        if self.root:
            self.root.quit()
            self.root.destroy()

    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
# gui_app.py

import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import queue
import cv2 as cv # Import cv2 for color conversion, just in case

# Correctly import the VisionBackend class from our refactored file
from vision_backend import VisionBackend 

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- App Window Configuration ---
        self.title("Professor Gesture Control")
        self.geometry("1024x680")
        ctk.set_appearance_mode("dark")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- State & Threading ---
        self.vision_thread = None
        self.stop_event = threading.Event()
        self.gui_queue = queue.Queue()
        self.vision_backend = None # Initialize to None
        
        # --- Widgets ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Video Display Label
        self.video_label = ctk.CTkLabel(self, text="Press 'Start' to begin camera feed", font=ctk.CTkFont(size=20))
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Control Frame
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.control_frame.grid_columnconfigure(2, weight=1) # Make status label expand

        self.start_button = ctk.CTkButton(self.control_frame, text="Start", command=self.start_vision_thread)
        self.start_button.grid(row=0, column=0, padx=5, pady=5)

        self.stop_button = ctk.CTkButton(self.control_frame, text="Stop", command=self.stop_vision_thread, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Status Label
        self.status_label = ctk.CTkLabel(self.control_frame, text="Status: Ready", anchor="w")
        self.status_label.grid(row=0, column=2, padx=10, sticky="ew")

        self.quit_button = ctk.CTkButton(self.control_frame, text="Quit", command=self.on_closing)
        self.quit_button.grid(row=0, column=3, padx=5, pady=5)
        
        # Start the queue polling loop
        self.after(100, self.poll_queue)

    def start_vision_thread(self):
        """Starts the background thread for video processing."""
        self.stop_event.clear()
        try:
            # Initialize the backend class
            self.vision_backend = VisionBackend(cap_device=0) 
            
            # Create and start the thread
            self.vision_thread = threading.Thread(
                target=self.vision_backend.run, 
                args=(self.gui_queue, self.stop_event),
                daemon=True # Daemon threads exit when the main program exits
            )
            self.vision_thread.start()
            
            # Update GUI state
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status_label.configure(text="Status: Starting vision thread...")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}")

    def stop_vision_thread(self):
        """Signals the background thread to stop."""
        if self.vision_thread and self.vision_thread.is_alive():
            self.stop_event.set()
            # The thread will clean up itself (cap.release())
            
            # Update GUI state
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_label.configure(text="Status: Stopping...")

    def poll_queue(self):
        """Periodically checks the queue for messages from the vision thread."""
        try:
            # Process all available messages in the queue to avoid lag
            while not self.gui_queue.empty():
                message = self.gui_queue.get_nowait()
                msg_type, data = message

                if msg_type == "image":
                    # Convert numpy array (OpenCV image) to CTkImage
                    img = Image.fromarray(data)
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(960, 540))
                    self.video_label.configure(image=ctk_img, text="")
                
                elif msg_type == "status":
                    self.status_label.configure(text=f"Status: {data}")
                
                elif msg_type == "action":
                    # Display actions temporarily
                    self.status_label.configure(text=f"Action: {data}")

        except queue.Empty:
            pass  # No new messages
        finally:
            # Continue polling
            self.after(100, self.poll_queue)
    
    def on_closing(self):
        """Called when the window is closed."""
        self.stop_vision_thread()
        # Give the thread a moment to stop before destroying the window
        self.after(200, self.destroy)

if __name__ == "__main__":
    app = App()
    app.mainloop()

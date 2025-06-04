#!/usr/bin/env python3
"""
Enhanced Hand Gesture Virtual Mouse & Keyboard System
Fixed version with proper constructor names and error handling
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import time
import json
import os
import platform
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox
import math

# Cross-platform imports
try:
    import pynput
    from pynput.mouse import Button, Listener as MouseListener
    from pynput.keyboard import Key, Listener as KeyboardListener
except ImportError:
    print("Warning: pynput not available. Some features may be limited.")
    pynput = None

# Disable pyautogui fail-safe for smooth operation
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

@dataclass
class GestureConfig:
    """Configuration class for gesture settings"""
    mouse_sensitivity: float = 2.0
    smoothing_factor: float = 0.7
    click_threshold: float = 0.03
    gesture_threshold: float = 0.1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    enable_virtual_keyboard: bool = True
    keyboard_layout: str = "qwerty"
    multi_hand_mode: bool = True

class HandGestureRecognizer:
    """Advanced hand gesture recognition system"""

    def __init__(self, config: GestureConfig):  # Fixed: was _init_
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize MediaPipe hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2 if config.multi_hand_mode else 1,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )

        # Gesture state tracking
        self.previous_positions = deque(maxlen=10)
        self.gesture_history = deque(maxlen=20)
        self.click_cooldown = False
        self.last_click_time = 0

        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Gesture definitions
        self.gesture_map = {
            'point': self._detect_point_gesture,
            'click': self._detect_click_gesture,
            'double_click': self._detect_double_click_gesture,
            'right_click': self._detect_right_click_gesture,
            'scroll_up': self._detect_scroll_up_gesture,
            'scroll_down': self._detect_scroll_down_gesture,
            'drag': self._detect_drag_gesture,
            'keyboard_mode': self._detect_keyboard_mode_gesture,
            'zoom_in': self._detect_zoom_in_gesture,
            'zoom_out': self._detect_zoom_out_gesture
        }

    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)  # Fixed: was *2

    def _detect_point_gesture(self, landmarks):
        """Detect pointing gesture (index finger extended)"""
        # Check if index finger is extended while others are folded
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]

        index_extended = index_tip.y < index_mcp.y
        middle_folded = middle_tip.y > middle_mcp.y

        return index_extended and middle_folded

    def _detect_click_gesture(self, landmarks):
        """Detect click gesture (thumb and index finger pinch)"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = self._calculate_distance(thumb_tip, index_tip)
        return distance < self.config.click_threshold

    def _detect_double_click_gesture(self, landmarks):
        """Detect double click gesture (rapid pinch motion)"""
        if self._detect_click_gesture(landmarks):
            current_time = time.time()
            if (current_time - self.last_click_time) < 0.5:
                return True
        return False

    def _detect_right_click_gesture(self, landmarks):
        """Detect right click gesture (thumb and middle finger pinch)"""
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        distance = self._calculate_distance(thumb_tip, middle_tip)
        return distance < self.config.click_threshold

    def _detect_scroll_up_gesture(self, landmarks):
        """Detect scroll up gesture (hand moving up with fist)"""
        # Implementation for fist detection and upward movement
        return False  # Placeholder

    def _detect_scroll_down_gesture(self, landmarks):
        """Detect scroll down gesture (hand moving down with fist)"""
        # Implementation for fist detection and downward movement
        return False  # Placeholder

    def _detect_drag_gesture(self, landmarks):
        """Detect drag gesture (sustained pinch while moving)"""
        return self._detect_click_gesture(landmarks)

    def _detect_keyboard_mode_gesture(self, landmarks):
        """Detect keyboard mode gesture (peace sign)"""
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]

        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y

        return index_extended and middle_extended

    def _detect_zoom_in_gesture(self, landmarks):
        """Detect zoom in gesture (two hands spreading apart)"""
        # This would require two-hand detection
        return False  # Placeholder

    def _detect_zoom_out_gesture(self, landmarks):
        """Detect zoom out gesture (two hands coming together)"""
        # This would require two-hand detection
        return False  # Placeholder

    def process_frame(self, frame):
        """Process video frame and detect gestures"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gestures_detected = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Detect gestures
                landmarks = hand_landmarks.landmark
                for gesture_name, detector in self.gesture_map.items():
                    if detector(landmarks):
                        gestures_detected.append(gesture_name)

                # Get cursor position from index finger tip
                if self._detect_point_gesture(landmarks):
                    index_tip = landmarks[8]
                    cursor_x = int(index_tip.x * self.screen_width)
                    cursor_y = int(index_tip.y * self.screen_height)

                    # Apply smoothing
                    if self.previous_positions:
                        prev_x, prev_y = self.previous_positions[-1]
                        cursor_x = int(prev_x + (cursor_x - prev_x) * self.config.smoothing_factor)
                        cursor_y = int(prev_y + (cursor_y - prev_y) * self.config.smoothing_factor)

                    self.previous_positions.append((cursor_x, cursor_y))

                    # Move mouse cursor
                    try:
                        pyautogui.moveTo(cursor_x, cursor_y)
                    except pyautogui.FailSafeException:
                        pass

        return frame, gestures_detected

class VirtualKeyboard:
    """Virtual keyboard implementation"""

    def __init__(self, layout="qwerty"):  # Fixed: was _init_
        self.layout = layout
        self.keyboards = {
            "qwerty": [
                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
                ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
                ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
                ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
                ['space', 'backspace', 'enter']
            ],
            "numeric": [
                ['1', '2', '3'],
                ['4', '5', '6'],
                ['7', '8', '9'],
                ['0', '.', 'enter']
            ]
        }

        self.current_keyboard = self.keyboards[layout]
        self.window = None
        self.is_visible = False

    def show_keyboard(self):
        """Display the virtual keyboard"""
        if self.window is None:
            self._create_keyboard_window()

        self.window.deiconify()
        self.window.lift()
        self.is_visible = True

    def hide_keyboard(self):
        """Hide the virtual keyboard"""
        if self.window:
            self.window.withdraw()
            self.is_visible = False

    def _create_keyboard_window(self):
        """Create the virtual keyboard window"""
        self.window = tk.Toplevel()
        self.window.title("Virtual Keyboard")
        self.window.configure(bg='#2c3e50')
        self.window.attributes('-topmost', True)

        # Position at bottom of screen
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        window_width = 700
        window_height = 350

        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 50

        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Create keyboard buttons
        for row_idx, row in enumerate(self.current_keyboard):
            frame = tk.Frame(self.window, bg='#2c3e50')
            frame.pack(pady=2)

            for key in row:
                btn_width = 8 if key not in ['space', 'backspace', 'enter'] else 15

                btn = tk.Button(
                    frame,
                    text=key.upper() if key not in ['space', 'backspace', 'enter'] else key.title(),
                    width=btn_width,
                    height=2,
                    font=('Arial', 10, 'bold'),
                    bg='#34495e',
                    fg='white',
                    activebackground='#3498db',
                    activeforeground='white',
                    command=lambda k=key: self._key_pressed(k)
                )
                btn.pack(side=tk.LEFT, padx=2)

        # Close button
        close_btn = tk.Button(
            self.window,
            text="Close Keyboard",
            command=self.hide_keyboard,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        close_btn.pack(pady=10)

        self.window.withdraw()  # Start hidden

    def _key_pressed(self, key):
        """Handle virtual key press"""
        if key == 'space':
            pyautogui.press('space')
        elif key == 'backspace':
            pyautogui.press('backspace')
        elif key == 'enter':
            pyautogui.press('enter')
        else:
            pyautogui.write(key)

class MouseController:
    """Enhanced mouse control with gesture integration"""

    def __init__(self, config: GestureConfig):  # Fixed: was _init_
        self.config = config
        self.dragging = False
        self.drag_start_pos = None

    def handle_gesture(self, gesture, position=None):
        """Handle detected gesture"""
        try:
            if gesture == 'click':
                pyautogui.click()
            elif gesture == 'double_click':
                pyautogui.doubleClick()
            elif gesture == 'right_click':
                pyautogui.rightClick()
            elif gesture == 'drag' and position:
                if not self.dragging:
                    self.dragging = True
                    self.drag_start_pos = position
                    pyautogui.mouseDown()
                else:
                    pyautogui.dragTo(position[0], position[1])
            elif gesture == 'scroll_up':
                pyautogui.scroll(3)
            elif gesture == 'scroll_down':
                pyautogui.scroll(-3)
            elif self.dragging and gesture != 'drag':
                # Stop dragging
                pyautogui.mouseUp()
                self.dragging = False
                self.drag_start_pos = None

        except Exception as e:
            print(f"Error handling gesture {gesture}: {e}")

class EnhancedGestureSystem:
    """Main system combining all components"""

    def __init__(self):  # Fixed: was _init_
        self.config = GestureConfig()
        self.gesture_recognizer = HandGestureRecognizer(self.config)
        self.virtual_keyboard = VirtualKeyboard(self.config.keyboard_layout)
        self.mouse_controller = MouseController(self.config)

        self.running = False
        self.camera = None
        self.gui_root = None

        # Performance monitoring
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0

    def start_camera(self):
        """Initialize and start camera"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Cannot access camera")

        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

    def stop_camera(self):
        """Stop and release camera"""
        if self.camera:
            self.camera.release()
            self.camera = None

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_timer >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_timer = current_time

    def create_gui(self):
        """Create the main GUI interface"""
        self.gui_root = tk.Tk()
        self.gui_root.title("Enhanced Hand Gesture Control System")
        self.gui_root.configure(bg='#2c3e50')
        self.gui_root.geometry("800x600")

        # Title
        title_label = tk.Label(
            self.gui_root,
            text="Enhanced Hand Gesture Virtual Mouse & Keyboard",
            font=('Arial', 16, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)

        # Control buttons frame
        control_frame = tk.Frame(self.gui_root, bg='#2c3e50')
        control_frame.pack(pady=20)

        # Start/Stop button
        self.start_btn = tk.Button(
            control_frame,
            text="Start System",
            command=self.toggle_system,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=15,
            height=2
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)

        # Keyboard button
        keyboard_btn = tk.Button(
            control_frame,
            text="Virtual Keyboard",
            command=self.toggle_keyboard,
            bg='#3498db',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=15,
            height=2
        )
        keyboard_btn.pack(side=tk.LEFT, padx=10)

        # Settings button
        settings_btn = tk.Button(
            control_frame,
            text="Settings",
            command=self.show_settings,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 12, 'bold'),
            width=15,
            height=2
        )
        settings_btn.pack(side=tk.LEFT, padx=10)

        # Status frame
        status_frame = tk.LabelFrame(
            self.gui_root,
            text="System Status",
            font=('Arial', 12, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        status_frame.pack(pady=20, padx=20, fill='x')

        self.status_label = tk.Label(
            status_frame,
            text="System: Stopped | FPS: 0 | Gestures: None",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='white'
        )
        self.status_label.pack(pady=10)

        # Gesture instructions
        instructions_frame = tk.LabelFrame(
            self.gui_root,
            text="Gesture Instructions",
            font=('Arial', 12, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        instructions_frame.pack(pady=20, padx=20, fill='both', expand=True)

        instructions_text = tk.Text(
            instructions_frame,
            bg='#34495e',
            fg='white',
            font=('Arial', 10),
            wrap=tk.WORD
        )
        instructions_text.pack(padx=10, pady=10, fill='both', expand=True)

        instructions = """
        GESTURE CONTROLS:
        
        🖱 MOUSE CONTROL:
        • Point (Index finger up) - Move cursor
        • Pinch (Thumb + Index) - Left click
        • Thumb + Middle finger - Right click
        • Double pinch quickly - Double click
        • Hold pinch and move - Drag
        
        ⌨ KEYBOARD:
        • Peace sign (Index + Middle up) - Toggle virtual keyboard
        • Use virtual keyboard for typing
        
        🎮 ADVANCED:
        • Two hands spread - Zoom in (if supported)
        • Two hands together - Zoom out (if supported)
        • Fist up/down - Scroll up/down (if supported)
        
        💡 TIPS:
        • Keep hand steady for accurate control
        • Ensure good lighting for better detection
        • Adjust sensitivity in settings if needed
        • Use smooth, deliberate movements
        """

        instructions_text.insert(tk.END, instructions)
        instructions_text.config(state=tk.DISABLED)

    def toggle_system(self):
        """Start or stop the gesture recognition system"""
        if not self.running:
            self.start_system()
        else:
            self.stop_system()

    def start_system(self):
        """Start the gesture recognition system"""
        try:
            self.start_camera()
            self.running = True
            self.start_btn.config(text="Stop System", bg='#e74c3c')

            # Start the main processing loop in a separate thread
            self.processing_thread = threading.Thread(target=self.processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system: {e}")

    def stop_system(self):
        """Stop the gesture recognition system"""
        self.running = False
        self.stop_camera()
        cv2.destroyAllWindows()
        self.start_btn.config(text="Start System", bg='#27ae60')

    def toggle_keyboard(self):
        """Toggle virtual keyboard visibility"""
        if self.virtual_keyboard.is_visible:
            self.virtual_keyboard.hide_keyboard()
        else:
            self.virtual_keyboard.show_keyboard()

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.gui_root)
        settings_window.title("Settings")
        settings_window.configure(bg='#2c3e50')
        settings_window.geometry("700x600")

        # Mouse sensitivity
        tk.Label(settings_window, text="Mouse Sensitivity:", bg='#2c3e50', fg='white').pack(pady=5)
        sensitivity_var = tk.DoubleVar(value=self.config.mouse_sensitivity)
        sensitivity_scale = tk.Scale(
            settings_window, from_=0.5, to=5.0, resolution=0.1,
            orient=tk.HORIZONTAL, variable=sensitivity_var,
            bg='#34495e', fg='white', highlightbackground='#2c3e50'
        )
        sensitivity_scale.pack(pady=5, padx=20, fill='x')

        # Click threshold
        tk.Label(settings_window, text="Click Threshold:", bg='#2c3e50', fg='white').pack(pady=5)
        click_var = tk.DoubleVar(value=self.config.click_threshold)
        click_scale = tk.Scale(
            settings_window, from_=0.01, to=0.1, resolution=0.01,
            orient=tk.HORIZONTAL, variable=click_var,
            bg='#34495e', fg='white', highlightbackground='#2c3e50'
        )
        click_scale.pack(pady=5, padx=20, fill='x')

        # Multi-hand mode
        multi_hand_var = tk.BooleanVar(value=self.config.multi_hand_mode)
        multi_hand_check = tk.Checkbutton(
            settings_window, text="Enable Multi-hand Mode",
            variable=multi_hand_var, bg='#2c3e50', fg='white',
            selectcolor='#34495e'
        )
        multi_hand_check.pack(pady=10)

        # Save button
        def save_settings():
            self.config.mouse_sensitivity = sensitivity_var.get()
            self.config.click_threshold = click_var.get()
            self.config.multi_hand_mode = multi_hand_var.get()
            settings_window.destroy()
            messagebox.showinfo("Settings", "Settings saved successfully!")

        save_btn = tk.Button(
            settings_window, text="Save Settings",
            command=save_settings, bg='#27ae60', fg='white'
        )
        save_btn.pack(pady=20)

    def processing_loop(self):
        """Main processing loop for gesture detection"""
        while self.running:
            if self.camera is None:
                break

            ret, frame = self.camera.read()
            if not ret:
                continue

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process gestures
            processed_frame, detected_gestures = self.gesture_recognizer.process_frame(frame)

            # Handle detected gestures
            for gesture in detected_gestures:
                if gesture == 'keyboard_mode':
                    self.virtual_keyboard.show_keyboard()
                else:
                    self.mouse_controller.handle_gesture(gesture)

            # Add FPS and status overlay
            self.update_fps()
            cv2.putText(processed_frame, f"FPS: {self.current_fps}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Gestures: {', '.join(detected_gestures) if detected_gestures else 'None'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Update GUI status
            if self.gui_root:
                status_text = f"System: Running | FPS: {self.current_fps} | Gestures: {', '.join(detected_gestures) if detected_gestures else 'None'}"
                self.status_label.config(text=status_text)

            # Display frame
            cv2.imshow('Enhanced Hand Gesture Control', processed_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_system()

    def run(self):
        """Run the complete system"""
        print("Starting Enhanced Hand Gesture Virtual Mouse & Keyboard System...")
        print("Platform:", platform.system())

        # Create and run GUI
        self.create_gui()

        # Handle window close
        def on_closing():
            self.stop_system()
            self.gui_root.destroy()

        self.gui_root.protocol("WM_DELETE_WINDOW", on_closing)
        self.gui_root.mainloop()

def main():
    """Main function to run the system"""
    try:
        system = EnhancedGestureSystem()
        system.run()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":  # Fixed: was _name_ and _main_
    main()
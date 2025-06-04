import difflib
import math
import re
import threading
import time
import tkinter as ttk
from collections import deque
from dataclasses import dataclass
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pyttsx3
import speech_recognition as sr
import wikipedia

# Disable pyautogui failsafe for smooth operation
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

@dataclass
class GestureConfig:
    """Enhanced configuration for gesture recognition"""
    click_threshold: float = 0.05
    double_click_threshold: float = 0.3
    scroll_threshold: float = 0.1
    drag_threshold: float = 0.03
    palm_threshold: float = 0.8
    smoothing_factor: float = 0.7
    gesture_hold_time: float = 0.5
    sensitivity: float = 1.0

class SmartTextPredictor:
    """Advanced text prediction and autocorrect system"""

    def __init__(self):
        self.common_words = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use", "what", "will", "with", "have", "this", "that", "they", "them", "been", "said", "each", "which", "their", "time", "would", "there", "could", "other", "after", "first", "well", "water", "about", "think", "right", "where", "might", "before", "great", "should", "still", "never", "again", "these", "those", "every", "being", "place", "while", "small", "found", "asked", "house", "world", "below", "asked", "going", "large", "until", "along", "shall", "being", "often", "earth", "began", "since", "study", "night", "light", "above", "paper", "parts", "young", "story", "point", "times", "heard", "whole", "white", "given", "means", "music", "miles", "thing", "today", "later", "using", "money", "lines", "order", "group", "among", "learn", "known", "space", "table", "early", "trees", "short"
        ]

        self.word_frequency = {}
        self.load_word_frequency()

    def load_word_frequency(self):
        """Load word frequency data"""
        for i, word in enumerate(self.common_words):
            self.word_frequency[word] = len(self.common_words) - i

    def get_suggestions(self, partial_word, num_suggestions=5):
        """Get word suggestions based on partial input"""
        if not partial_word:
            return []

        partial_lower = partial_word.lower()
        suggestions = []

        # Exact prefix matches first
        for word in self.common_words:
            if word.startswith(partial_lower):
                suggestions.append(word)

        # Fuzzy matches for typos
        if len(suggestions) < num_suggestions:
            fuzzy_matches = difflib.get_close_matches(
                partial_lower, self.common_words,
                n=num_suggestions-len(suggestions), cutoff=0.6
            )
            suggestions.extend(fuzzy_matches)

        return suggestions[:num_suggestions]

    def autocorrect_word(self, word):
        if not word or len(word) < 2:
            return word

        word_lower = word.lower()

        # Check if word exists as-is
        if word_lower in self.common_words:
            return word


        matches = difflib.get_close_matches(word_lower, self.common_words, n=1, cutoff=0.8)
        if matches:
            return matches[0]

        return word

    def process_sentence(self, sentence):
        """Process and autocorrect entire sentence"""
        if not sentence:
            return sentence

        words = sentence.split()
        corrected_words = []

        for word in words:
            # Keep punctuation
            punctuation = ""
            clean_word = word

            if word and word[-1] in ".,!?;:":
                punctuation = word[-1]
                clean_word = word[:-1]

            corrected_word = self.autocorrect_word(clean_word)
            corrected_words.append(corrected_word + punctuation)

        return " ".join(corrected_words)

class VoiceAssistant:
    """Google Assistant-like voice command system"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
        except:
            self.microphone = None
            print("No microphone detected")

        try:
            self.tts = pyttsx3.init()
        except:
            self.tts = None
            print("Text-to-speech not available")

        self.is_listening = False
        self.wake_word = "hey computer"
        self.setup_voice()

        self.commands = {
            "search": self.search_web,
            "weather": self.get_weather,
            "time": self.get_time,
            "date": self.get_date,
            "calculator": self.calculate,
            "wikipedia": self.search_wikipedia,
            "translate": self.translate_text,
            "reminder": self.set_reminder,
            "mouse": self.mouse_commands,
            "keyboard": self.keyboard_commands,
            "system": self.system_commands,
        }

    def setup_voice(self):
        """Setup text-to-speech voice"""
        if not self.tts:
            return

        try:
            voices = self.tts.getProperty('voices')
            if voices:
                # Try to set a female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts.setProperty('voice', voice.id)
                        break

            self.tts.setProperty('rate', 180)
            self.tts.setProperty('volume', 0.8)
        except:
            pass

    def speak(self, text):
        """Text-to-speech with threading"""
        if not self.tts:
            print(f"Assistant: {text}")
            return

        def speak_thread():
            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except:
                print(f"Assistant: {text}")

        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()

    def search_web(self, query):
        """Search the web"""
        self.speak(f"Searching for {query}")
        try:
            import webbrowser
            webbrowser.open(f"https://www.google.com/search?q={query}")
        except:
            print(f"Would search for: {query}")

    def get_weather(self, location=""):
        """Get weather information"""
        self.speak("I would check the weather for you, but I need an API key for weather services.")

    def get_time(self):
        """Get current time"""
        current_time = time.strftime("%I:%M %p")
        self.speak(f"The current time is {current_time}")

    def get_date(self):
        """Get current date"""
        current_date = time.strftime("%B %d, %Y")
        self.speak(f"Today is {current_date}")

    def calculate(self, expression):
        """Perform calculations"""
        try:
            # Safe evaluation of basic math expressions
            allowed_chars = "0123456789+-*/.() "
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                self.speak(f"The result is {result}")
            else:
                self.speak("I can only do basic math calculations")
        except:
            self.speak("I couldn't calculate that expression")

    def search_wikipedia(self, query):
        """Search Wikipedia"""
        try:
            summary = wikipedia.summary(query, sentences=2)
            self.speak(summary)
        except:
            self.speak(f"I couldn't find information about {query} on Wikipedia")

    def translate_text(self, text):
        """Translate text (placeholder)"""
        self.speak("Translation feature would require an external API")

    def set_reminder(self, reminder):
        """Set a reminder (placeholder)"""
        self.speak(f"I've noted your reminder: {reminder}")

    def mouse_commands(self, command):
        """Handle mouse-related voice commands"""
        if "click" in command:
            pyautogui.click()
            self.speak("Clicked")
        elif "right click" in command:
            pyautogui.rightClick()
            self.speak("Right clicked")
        elif "double click" in command:
            pyautogui.doubleClick()
            self.speak("Double clicked")
        elif "scroll up" in command:
            pyautogui.scroll(3)
            self.speak("Scrolled up")
        elif "scroll down" in command:
            pyautogui.scroll(-3)
            self.speak("Scrolled down")

    def keyboard_commands(self, command):
        """Handle keyboard-related voice commands"""
        if "type" in command:
            text = command.replace("type", "").strip()
            pyautogui.write(text)
            self.speak(f"Typed: {text}")
        elif "enter" in command:
            pyautogui.press('enter')
            self.speak("Pressed Enter")
        elif "backspace" in command:
            pyautogui.press('backspace')
            self.speak("Pressed Backspace")

    def system_commands(self, command):
        """Handle system-related voice commands"""
        if "screenshot" in command:
            screenshot = pyautogui.screenshot()
            screenshot.save("screenshot.png")
            self.speak("Screenshot saved")

    def process_command(self, command_text):
        """Process voice command using NLP-like matching"""
        command_lower = command_text.lower()

        # Direct command matching
        for cmd_key, cmd_func in self.commands.items():
            if cmd_key in command_lower:
                if cmd_key in ["search", "wikipedia", "translate", "reminder"]:
                    query = command_lower.replace(cmd_key, "").strip()
                    if cmd_key == "search":
                        query = query.replace("for", "").strip()
                    cmd_func(query)
                elif cmd_key in ["mouse", "keyboard", "system"]:
                    cmd_func(command_lower)
                else:
                    cmd_func()
                return True

        # Math expressions
        if any(op in command_lower for op in ["plus", "minus", "times", "divided"]):
            math_expr = self.convert_speech_to_math(command_lower)
            if math_expr:
                self.calculate(math_expr)
                return True

        # Default response
        self.speak("I'm not sure how to help with that. Try saying 'search', 'weather', 'time', or 'calculator'")
        return False

    def convert_speech_to_math(self, speech):
        """Convert speech to mathematical expression"""
        speech = speech.replace("plus", "+")
        speech = speech.replace("minus", "-")
        speech = speech.replace("times", "*")
        speech = speech.replace("multiplied by", "*")
        speech = speech.replace("divided by", "/")

        # Extract numbers and operators
        pattern = r'[\d+\-*/\.\s]+'
        match = re.search(pattern, speech)
        if match:
            return match.group().strip()
        return None

    def listen_for_wake_word(self):
        """Listen for wake word continuously"""
        if not self.microphone:
            return False

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)

            text = self.recognizer.recognize_google(audio).lower()

            if self.wake_word in text:
                self.speak("Yes, how can I help you?")
                return self.listen_for_command()

        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except:
            pass

        return False

    def listen_for_command(self):
        """Listen for actual command after wake word"""
        if not self.microphone:
            return False

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

            command = self.recognizer.recognize_google(audio)
            print(f"Voice Command: {command}")
            return self.process_command(command)

        except sr.WaitTimeoutError:
            self.speak("I didn't hear anything")
        except sr.UnknownValueError:
            self.speak("I didn't understand that")
        except:
            self.speak("There was an error processing your command")

        return False

class AdvancedGestureRecognizer:
    """Enhanced gesture recognition with palm detection and advanced gestures"""

    def __init__(self, config: GestureConfig):
        self.config = config
        self.gesture_history = deque(maxlen=10)
        self.last_click_time = 0
        self.click_count = 0
        self.is_palm_open = False
        self.palm_hold_start = 0
        self.last_gesture_time = time.time()

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_finger_extended(self, landmarks, finger_tips, finger_pips):
        """Check if finger is extended"""
        tip_y = landmarks[finger_tips][1]
        pip_y = landmarks[finger_pips][1]
        return tip_y < pip_y

    def detect_open_palm(self, landmarks):
        """Detect open palm gesture for screen hold"""
        if landmarks is None:
            return False

        # Check if all fingers are extended
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
        finger_pips = [6, 10, 14, 18]

        extended_fingers = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if self.is_finger_extended(landmarks, tip, pip):
                extended_fingers += 1

        # Also check thumb
        thumb_extended = landmarks[4][0] > landmarks[3][0]  # Thumb tip vs thumb joint

        return extended_fingers >= 4 and thumb_extended

    def detect_closed_fist(self, landmarks):
        """Detect closed fist for releasing hold"""
        if landmarks is None:
            return False

        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        closed_fingers = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if not self.is_finger_extended(landmarks, tip, pip):
                closed_fingers += 1

        return closed_fingers >= 3

    def detect_click_gesture(self, landmarks):
        """Enhanced click detection with double-click support"""
        if landmarks is None:
            return {"single": False, "double": False, "right": False}

        thumb_tip = landmarks[4][:2]
        index_tip = landmarks[8][:2]
        distance = self.calculate_distance(thumb_tip, index_tip)

        current_time = time.time()
        is_clicking = distance < self.config.click_threshold

        result = {"single": False, "double": False, "right": False}

        if is_clicking:
            if current_time - self.last_click_time < self.config.double_click_threshold:
                self.click_count += 1
            else:
                self.click_count = 1

            self.last_click_time = current_time

            # Check for right click (thumb + middle finger)
            middle_tip = landmarks[12][:2]
            thumb_middle_distance = self.calculate_distance(thumb_tip, middle_tip)

            if thumb_middle_distance < self.config.click_threshold:
                result["right"] = True
            elif self.click_count == 2:
                result["double"] = True
                self.click_count = 0
            else:
                result["single"] = True

        return result

    def detect_scroll_gesture(self, landmarks):
        """Enhanced scrolling with directional control"""
        if landmarks is None:
            return {"direction": None, "intensity": 0}

        index_tip = landmarks[8][:2]
        middle_tip = landmarks[12][:2]

        # Check if index and middle fingers are extended
        index_extended = self.is_finger_extended(landmarks, 8, 6)
        middle_extended = self.is_finger_extended(landmarks, 12, 10)

        # Check if ring and pinky are closed
        ring_closed = not self.is_finger_extended(landmarks, 16, 14)
        pinky_closed = not self.is_finger_extended(landmarks, 20, 18)

        if index_extended and middle_extended and ring_closed and pinky_closed:
            # Calculate scroll direction and intensity
            center_y = (index_tip[1] + middle_tip[1]) / 2

            # Get movement intensity based on finger position
            wrist_y = landmarks[0][1]
            relative_position = center_y - wrist_y

            if relative_position < -0.1:  # Fingers pointing up
                return {"direction": "up", "intensity": abs(relative_position) * 5}
            elif relative_position > 0.1:  # Fingers pointing down
                return {"direction": "down", "intensity": abs(relative_position) * 5}

        return {"direction": None, "intensity": 0}

class EnhancedVirtualKeyboard:
    """Enhanced virtual keyboard with smart text prediction"""

    def __init__(self):
        self.layout = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['SPACE', 'BACKSPACE', 'ENTER']
        ]

        self.current_text = ""
        self.current_word = ""
        self.text_predictor = SmartTextPredictor()
        self.suggestions = []
        self.is_active = False
        self.last_key_time = 0

    def get_key_at_position(self, x, y, keyboard_region):
        """Get keyboard key at position with improved accuracy"""
        if not keyboard_region:
            return None

        kx, ky, kw, kh = keyboard_region
        if not (kx <= x <= kx + kw and ky <= y <= ky + kh):
            return None

        rel_x = (x - kx) / kw
        rel_y = (y - ky) / kh

        # Determine row (accounting for different row heights)
        if rel_y < 0.6:  # Regular letter rows
            row_idx = int(rel_y * 3)
            if row_idx >= 3:
                row_idx = 2
        else:  # Special keys row
            row_idx = 3

        if row_idx >= len(self.layout):
            return None

        row = self.layout[row_idx]

        if row_idx == 3:  # Special keys
            if rel_x < 0.4:
                return 'SPACE'
            elif rel_x < 0.7:
                return 'BACKSPACE'
            else:
                return 'ENTER'
        else:
            col_idx = int(rel_x * len(row))
            if col_idx >= len(row):
                col_idx = len(row) - 1

            return row[col_idx]

    def type_key(self, key):
        """Enhanced key typing with smart predictions"""
        current_time = time.time()

        # Prevent rapid repeated key presses
        if current_time - self.last_key_time < 0.3:
            return

        self.last_key_time = current_time

        if key == 'SPACE':
            if self.current_word:
                corrected_word = self.text_predictor.autocorrect_word(self.current_word)
                # Replace the current word with corrected version
                if corrected_word != self.current_word.lower():
                    # Remove current word and add corrected word
                    for _ in range(len(self.current_word)):
                        pyautogui.press('backspace')
                    pyautogui.write(corrected_word)

                self.current_text = self.current_text[:-len(self.current_word)] + corrected_word
                self.current_word = ""

            pyautogui.press('space')
            self.current_text += " "

        elif key == 'BACKSPACE':
            pyautogui.press('backspace')
            if self.current_text:
                if self.current_text[-1] == ' ':
                    # Restore the word we were typing
                    words = self.current_text.strip().split()
                    if words:
                        self.current_word = words[-1]
                else:
                    if self.current_word:
                        self.current_word = self.current_word[:-1]

                self.current_text = self.current_text[:-1]

        elif key == 'ENTER':
            # Auto-correct entire sentence before entering
            corrected_text = self.text_predictor.process_sentence(self.current_text + self.current_word)

            # Clear current line and type corrected text
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.write(corrected_text)
            pyautogui.press('enter')

            self.current_text = ""
            self.current_word = ""

        else:
            # Regular letter
            pyautogui.write(key.lower())
            self.current_word += key.lower()
            self.current_text += key.lower()

        self.update_suggestions()

    def update_suggestions(self):
        """Update word suggestions"""
        if self.current_word:
            self.suggestions = self.text_predictor.get_suggestions(self.current_word)
        else:
            self.suggestions = []

    def apply_suggestion(self, suggestion_idx):
        """Apply a word suggestion"""
        if 0 <= suggestion_idx < len(self.suggestions):
            suggestion = self.suggestions[suggestion_idx]

            # Remove current word
            for _ in range(len(self.current_word)):
                pyautogui.press('backspace')

            # Type suggestion
            pyautogui.write(suggestion)

            # Update internal state
            self.current_text = self.current_text[:-len(self.current_word)] + suggestion
            self.current_word = ""
            self.suggestions = []

class AdvancedGestureSystem:
    """Enhanced main gesture control system"""

    def __init__(self):
        self.config = GestureConfig()
        self.hand_tracker = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.gesture_recognizer = AdvancedGestureRecognizer(self.config)
        self.virtual_keyboard = EnhancedVirtualKeyboard()
        self.voice_assistant = VoiceAssistant()

        self.cap = None
        self.is_running = False
        self.show_keyboard = False
        self.screen_hold = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        self.prev_cursor_x, self.prev_cursor_y = 0, 0

    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return self.cap.isOpened()

    def get_landmarks(self, results, hand_idx=0):
        """Extract landmarks from MediaPipe results"""
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_idx:
            landmarks = []
            for lm in results.multi_hand_landmarks[hand_idx].landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            return np.array(landmarks)
        return None

    def smooth_cursor_movement(self, x, y):
        """Apply smoothing to cursor movement"""
        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height)

        smooth_x = self.prev_cursor_x + (screen_x - self.prev_cursor_x) * (1 - self.config.smoothing_factor)
        smooth_y = self.prev_cursor_y + (screen_y - self.prev_cursor_y) * (1 - self.config.smoothing_factor)

        self.prev_cursor_x, self.prev_cursor_y = smooth_x, smooth_y
        return smooth_x, smooth_y

    def process_gestures(self, landmarks):
        """Enhanced gesture processing with all new features"""
        if landmarks is None:
            return

        # Palm detection for screen hold/release
        if self.gesture_recognizer.detect_open_palm(landmarks):
            if not self.screen_hold:
                self.screen_hold = True
                print("Screen Hold Activated")
        elif self.gesture_recognizer.detect_closed_fist(landmarks):
            if self.screen_hold:
                self.screen_hold = False
                print("Screen Hold Released")

        # Skip cursor movement if screen is held
        if not self.screen_hold and self.is_running:
            # Cursor movement using index finger
            index_tip = landmarks[8]
            smooth_x, smooth_y = self.smooth_cursor_movement(index_tip[0], index_tip[1])
            pyautogui.moveTo(smooth_x, smooth_y)

        if not self.is_running:
            return

        # Click detection (single, double, right-click)
        click_result = self.gesture_recognizer.detect_click_gesture(landmarks)

        if click_result["single"]:
            if self.show_keyboard:
                # Keyboard interaction
                key = self.virtual_keyboard.get_key_at_position(
                    landmarks[8][0], landmarks[8][1],
                    (0.05, 0.5, 0.9, 0.45)  # Keyboard region
                )
                if key:
                    self.virtual_keyboard.type_key(key)
            else:
                pyautogui.click()

        elif click_result["double"]:
            pyautogui.doubleClick()
            print("Double Click")

        elif click_result["right"]:
            pyautogui.rightClick()
            print("Right Click")

        # Scrolling with enhanced control
        scroll_result = self.gesture_recognizer.detect_scroll_gesture(landmarks)
        if scroll_result["direction"]:
            scroll_amount = int(scroll_result["intensity"])
            if scroll_result["direction"] == "up":
                pyautogui.scroll(scroll_amount)
            else:
                pyautogui.scroll(-scroll_amount)

    def draw_ui(self, frame):
        """Enhanced UI with more information"""
        height, width = frame.shape[:2]

        # Main status panel
        panel_height = 150
        cv2.rectangle(frame, (10, 10), (450, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, panel_height), (255, 255, 255), 2)

        # Status information
        status_info = [
            f"FPS: {self.fps:.1f}",
            f"System: {'Active' if self.is_running else 'Inactive'}",
            f"Screen Hold: {'ON' if self.screen_hold else 'OFF'}",
            f"Keyboard: {'ON' if self.show_keyboard else 'OFF'}",
            f"Voice Assistant: {'ON' if self.voice_assistant.is_listening else 'OFF'}",
        ]

        for i, info in enumerate(status_info):
            color = (0, 255, 0) if ('Active' in info or 'ON' in info) else (255, 255, 255)
            if 'Inactive' in info or 'OFF' in info:
                color = (0, 0, 255)
            cv2.putText(frame, info, (20, 40 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Controls help
        help_y = panel_height + 30
        cv2.putText(frame, "Controls:", (20, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, "Space: Toggle System | K: Keyboard | V: Voice | Q: Quit",
                    (20, help_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Gesture instructions
        gesture_help = [
            "Open Palm = Hold Screen",
            "Closed Fist = Release Hold",
            "Thumb+Index = Click",
            "Thumb+Middle = Right Click",
            "Two Fingers = Scroll"
        ]

        for i, instruction in enumerate(gesture_help):
            cv2.putText(frame, instruction, (20, help_y + 55 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Virtual keyboard
        if self.show_keyboard and self.virtual_keyboard:
            self.draw_virtual_keyboard(frame)

        # Text predictions
        if self.virtual_keyboard and self.virtual_keyboard.suggestions:
            self.draw_suggestions(frame)

    def draw_virtual_keyboard(self, frame):
        """Draw enhanced virtual keyboard"""
        height, width = frame.shape[:2]

        # Keyboard background
        kb_x, kb_y = int(width * 0.05), int(height * 0.5)
        kb_w, kb_h = int(width * 0.9), int(height * 0.45)

        cv2.rectangle(frame, (kb_x, kb_y), (kb_x + kb_w, kb_y + kb_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (kb_x, kb_y), (kb_x + kb_w, kb_y + kb_h), (255, 255, 255), 2)

        # Draw keys
        for row_idx, row in enumerate(self.virtual_keyboard.layout):
            if row_idx < 3:  # Regular letter rows
                key_width = kb_w // len(row)
                key_height = int(kb_h * 0.6 / 3)

                for col_idx, key in enumerate(row):
                    x = kb_x + col_idx * key_width
                    y = kb_y + row_idx * key_height

                    # Key background
                    cv2.rectangle(frame, (x + 2, y + 2), (x + key_width - 2, y + key_height - 2),
                                  (100, 100, 100), -1)
                    cv2.rectangle(frame, (x + 2, y + 2), (x + key_width - 2, y + key_height - 2),
                                  (200, 200, 200), 1)

                    # Key text
                    text_x = x + key_width // 2 - 10
                    text_y = y + key_height // 2 + 5
                    cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1)

            else:  # Special keys row
                special_keys = ['SPACE', 'BACKSPACE', 'ENTER']
                key_widths = [int(kb_w * 0.4), int(kb_w * 0.3), int(kb_w * 0.3)]

                x_offset = kb_x
                for i, (key, key_width) in enumerate(zip(special_keys, key_widths)):
                    y = kb_y + int(kb_h * 0.6)
                    key_height = int(kb_h * 0.4)

                    cv2.rectangle(frame, (x_offset + 2, y + 2),
                                  (x_offset + key_width - 2, y + key_height - 2),
                                  (100, 100, 100), -1)
                    cv2.rectangle(frame, (x_offset + 2, y + 2),
                                  (x_offset + key_width - 2, y + key_height - 2),
                                  (200, 200, 200), 1)

                    text_x = x_offset + key_width // 2 - len(key) * 4
                    text_y = y + key_height // 2 + 5
                    cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)

                    x_offset += key_width

        # Current text display
        text_area_y = kb_y - 40
        cv2.rectangle(frame, (kb_x, text_area_y), (kb_x + kb_w, kb_y - 5), (30, 30, 30), -1)
        cv2.rectangle(frame, (kb_x, text_area_y), (kb_x + kb_w, kb_y - 5), (255, 255, 255), 1)

        display_text = self.virtual_keyboard.current_text + self.virtual_keyboard.current_word
        if len(display_text) > 50:
            display_text = "..." + display_text[-47:]

        cv2.putText(frame, display_text, (kb_x + 10, text_area_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_suggestions(self, frame):
        """Draw word suggestions"""
        height, width = frame.shape[:2]

        # Suggestions panel
        sugg_x = int(width * 0.05)
        sugg_y = int(height * 0.35)
        sugg_w = int(width * 0.9)
        sugg_h = 40

        cv2.rectangle(frame, (sugg_x, sugg_y), (sugg_x + sugg_w, sugg_y + sugg_h),
                      (0, 50, 100), -1)
        cv2.rectangle(frame, (sugg_x, sugg_y), (sugg_x + sugg_w, sugg_y + sugg_h),
                      (255, 255, 255), 1)

        cv2.putText(frame, "Suggestions:", (sugg_x + 10, sugg_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw suggestions
        x_offset = sugg_x + 100
        for i, suggestion in enumerate(self.virtual_keyboard.suggestions[:5]):
            cv2.putText(frame, f"{i+1}. {suggestion}", (x_offset, sugg_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            x_offset += len(suggestion) * 8 + 40

    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def voice_thread(self):
        """Voice assistant thread"""
        if not self.voice_assistant:
            print("Voice assistant not available")
            return

        while self.cap and self.cap.isOpened():
            try:
                if hasattr(self.voice_assistant, 'is_listening') and self.voice_assistant.is_listening:
                    self.voice_assistant.listen_for_wake_word()
                time.sleep(0.1)
            except Exception as e:
                print(f"Voice thread error: {e}")
                time.sleep(1)

    def run(self):
        """Main execution loop"""
        if not self.initialize_camera():
            print("Error: Could not initialize camera")
            return

        if not self.hand_tracker:
            print("Error: MediaPipe not initialized")
            return

        print("Advanced Hand Gesture System Started")
        print("Controls:")
        print("  SPACE: Toggle gesture recognition")
        print("  K: Toggle virtual keyboard")
        print("  V: Toggle voice assistant")
        print("  Q: Quit")
        print("\nGestures:")
        print("  Open Palm: Hold/freeze screen")
        print("  Closed Fist: Release screen hold")
        print("  Thumb + Index: Click")
        print("  Thumb + Middle: Right click")
        print("  Two fingers up/down: Scroll")

        # Start voice assistant thread
        if self.voice_assistant:
            voice_thread = threading.Thread(target=self.voice_thread, daemon=True)
            voice_thread.start()

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading from camera")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                try:
                    results = self.hand_tracker.process(rgb_frame)
                except Exception as e:
                    print(f"MediaPipe processing error: {e}")
                    results = None

                # Process hand landmarks
                if results and results.multi_hand_landmarks:
                    # Draw hand landmarks
                    for hand_landmarks in results.multi_hand_landmarks:
                        if hasattr(self, 'mp_drawing') and hasattr(self, 'mp_drawing_styles'):
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                    # Get landmarks and process gestures
                    if self.gesture_recognizer:
                        landmarks = self.get_landmarks(results)
                        self.process_gestures(landmarks)

                # Calculate FPS
                self.calculate_fps()

                # Draw UI
                self.draw_ui(frame)

                # Display frame
                cv2.imshow('Advanced Hand Gesture Control', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.is_running = not self.is_running
                    print(f"Gesture Recognition: {'ON' if self.is_running else 'OFF'}")
                elif key == ord('k'):
                    self.show_keyboard = not self.show_keyboard
                    print(f"Virtual Keyboard: {'ON' if self.show_keyboard else 'OFF'}")
                elif key == ord('v'):
                    if self.voice_assistant:
                        if not hasattr(self.voice_assistant, 'is_listening'):
                            self.voice_assistant.is_listening = False
                        self.voice_assistant.is_listening = not self.voice_assistant.is_listening
                        print(f"Voice Assistant: {'ON' if self.voice_assistant.is_listening else 'OFF'}")
                    else:
                        print("Voice Assistant not available")
                elif key == ord('r'):
                    # Reset system
                    self.screen_hold = False
                    self.is_running = False
                    self.show_keyboard = False
                    print("System Reset")
                elif key >= ord('1') and key <= ord('5') and self.virtual_keyboard and self.virtual_keyboard.suggestions:
                    # Apply suggestion
                    suggestion_idx = key - ord('1')
                    self.virtual_keyboard.apply_suggestion(suggestion_idx)

        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Runtime error: {e}")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("System cleaned up successfully")

def create_settings_gui():
    """Create settings GUI for configuration"""
    def update_settings():
        try:
            config = GestureConfig(
                click_threshold=float(click_threshold_var.get()),
                double_click_threshold=float(double_click_var.get()),
                scroll_threshold=float(scroll_threshold_var.get()),
                smoothing_factor=float(smoothing_var.get()),
                sensitivity=float(sensitivity_var.get())
            )
            print("Settings updated successfully")
            root.destroy()

            # Start the main system with new config
            system = AdvancedGestureSystem()
            system.config = config
            system.run()

        except ValueError as e:
            print(f"Invalid setting value: {e}")

    root = tk.Tk()
    root.title("Advanced Gesture System Settings")
    root.geometry("400x300")

    # Settings variables
    click_threshold_var = tk.StringVar(value="0.05")
    double_click_var = tk.StringVar(value="0.3")
    scroll_threshold_var = tk.StringVar(value="0.1")
    smoothing_var = tk.StringVar(value="0.7")
    sensitivity_var = tk.StringVar(value="1.0")

    # Create settings UI
    settings_frame = ttk.Frame(root, padding="10")
    settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Click threshold
    ttk.Label(settings_frame, text="Click Threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
    ttk.Entry(settings_frame, textvariable=click_threshold_var, width=10).grid(row=0, column=1, pady=5)

    # Double click threshold
    ttk.Label(settings_frame, text="Double Click Threshold:").grid(row=1, column=0, sticky=tk.W, pady=5)
    ttk.Entry(settings_frame, textvariable=double_click_var, width=10).grid(row=1, column=1, pady=5)

    # Scroll threshold
    ttk.Label(settings_frame, text="Scroll Threshold:").grid(row=2, column=0, sticky=tk.W, pady=5)
    ttk.Entry(settings_frame, textvariable=scroll_threshold_var, width=10).grid(row=2, column=1, pady=5)

    # Smoothing factor
    ttk.Label(settings_frame, text="Smoothing Factor:").grid(row=3, column=0, sticky=tk.W, pady=5)
    ttk.Entry(settings_frame, textvariable=smoothing_var, width=10).grid(row=3, column=1, pady=5)

    # Sensitivity
    ttk.Label(settings_frame, text="Sensitivity:").grid(row=4, column=0, sticky=tk.W, pady=5)
    ttk.Entry(settings_frame, textvariable=sensitivity_var, width=10).grid(row=4, column=1, pady=5)

    # Buttons
    button_frame = ttk.Frame(settings_frame)
    button_frame.grid(row=5, column=0, columnspan=2, pady=20)

    ttk.Button(button_frame, text="Start System", command=update_settings).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=root.destroy).pack(side=tk.LEFT, padx=5)

    # Help text
    help_text = """
    Settings Help:
    - Click Threshold: Distance for click detection (lower = more sensitive)
    - Double Click Threshold: Time window for double clicks (seconds)
    - Scroll Threshold: Sensitivity for scroll gestures
    - Smoothing Factor: Cursor smoothing (0-1, higher = smoother)
    - Sensitivity: Overall gesture sensitivity multiplier
    """

    help_label = ttk.Label(settings_frame, text=help_text, font=("Arial", 8))
    help_label.grid(row=6, column=0, columnspan=2, pady=10, sticky=tk.W)

    root.mainloop()

def main():
    """Main entry point"""
    print("Advanced Hand Gesture Virtual Mouse & Keyboard System")
    print("====================================================")
    print()
    print("Features:")
    print("- Open palm screen hold/freeze")
    print("- Closed fist to release hold")
    print("- Advanced scrolling with finger gestures")
    print("- Double-click and right-click gestures")
    print("- Smart text prediction and autocorrect")
    print("- Google Assistant-like voice commands")
    print("- Virtual keyboard with suggestions")
    print()

    choice = input("Start with settings GUI? (y/n): ").lower().strip()

    if choice == 'y':
        create_settings_gui()
    else:
        system = AdvancedGestureSystem()
        system.run()

if __name__ == "__main__":
    main()


import random
import time
import speech_recognition as sr
import mediapipe as mp
import cv2
import numpy as np
from difflib import SequenceMatcher
from model_utils import load_model, predict_difficulty

class ListenAgent:
    """Real Agent for Speech Analysis using SpeechRecognition."""
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen_from_file(self, audio_file_path, target_text):
        """Analyzes audio file (wav) for accuracy and WPM."""
        print("Listen Agent: Processing audio file...")
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                try:
                    transcribed_text = self.recognizer.recognize_google(audio_data)
                    print(f"Listen Agent Output: '{transcribed_text}'")
                except sr.UnknownValueError:
                    print("Listen Agent: Could not understand audio")
                    return 1.0, 0 # Max error, 0 WPM
                except sr.RequestError as e:
                    print(f"Listen Agent: Service error; {e}")
                    return 0.5, 0 # Fallback

            # Calculate Error Rate
            seq = SequenceMatcher(None, target_text.lower(), transcribed_text.lower())
            accuracy = seq.ratio()
            error_rate = 1.0 - accuracy

            # Calculate WPM (Approximate based on file duration is hard without duration metadata from float)
            # Simple approximation: Word count / (Assumed typical speaking rate or just word count if instant)
            # Better: Pass duration. For now, let's use word count * 60 / duration if we had it.
            # Fallback: Just return raw word count as a proxy for speed if duration unknown, 
            # OR assume standard reading.
            # Let's count words.
            word_count = len(transcribed_text.split())
            # Rough fix: assume 5 seconds per sentence for this demo context if real time not available
            wpm = word_count * 12 # 60/5 = 12 multiplier
            
            return error_rate, wpm

        except Exception as e:
            print(f"Listen Agent Error: {e}")
            return 1.0, 0

class ObserveAgent:
    """Real Agent for Engagement Analysis using MediaPipe."""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def analyze_image(self, image_array):
        """Analyzes a single frame for engagement (focus)."""
        print("Observe Agent: Analyzing face...")
        try:
            results = self.face_mesh.process(image_array)
            
            if not results.multi_face_landmarks:
                print("Observe Agent: No face detected.")
                return 0.0 # No focus
            
            # Heuristic: If face is detected, we assume some focus. 
            # Deeper analysis would check head rotation (Pose).
            # Let's imply: Face Present = High Focus (0.8 - 1.0)
            # We can vary it slightly based on how 'centered' the nose is.
            
            landmarks = results.multi_face_landmarks[0]
            nose_tip = landmarks.landmark[4] # Index 4 is nose tip
            
            # Center is 0.5, 0.5. Calculate distance from center.
            dist_x = abs(nose_tip.x - 0.5)
            dist_y = abs(nose_tip.y - 0.5)
            deviation = np.sqrt(dist_x**2 + dist_y**2)
            
            # Max deviation roughly 0.5. Normalize focus.
            # Focus = 1.0 - (deviation * 2)
            focus_score = max(0.0, 1.0 - (deviation * 2.0))
            
            print(f"Observe Agent: Face found. Deviation: {deviation:.2f}. Focus: {focus_score:.2f}")
            return focus_score

        except Exception as e:
            print(f"Observe Agent Error: {e}")
            return 0.0

class AssistAgent:
    """Agent for Multisensory Learning Support."""
    def provide_assistance(self, difficulty_index):
        """Determines visual cues and audio support based on difficulty."""
        # ... (Same logic as before, just kept for completeness)
        assistance = {
            "visual_cues": [],
            "tts_enabled": False,
            "highlight_color": "none"
        }
        if difficulty_index > 0.7:
            assistance["visual_cues"] = ["syllable_segmentation", "font_spacing_wide"]
            assistance["tts_enabled"] = True
            assistance["highlight_color"] = "yellow"
        elif difficulty_index > 0.4:
            assistance["visual_cues"] = ["font_spacing_medium"]
            assistance["highlight_color"] = "light_blue"
        return assistance

class MentorAgent:
    """Personalized AI Feedback Coach (Template-based)."""
    def provide_feedback(self, error_rate, focus_score):
        """Generates personalized feedback using sophisticated templates."""
        print("Mentor Agent: Formulating feedback (Template Mode)...")
        
        # Determine State
        accuracy_state = "high" if error_rate < 0.2 else "medium" if error_rate < 0.5 else "low"
        focus_state = "high" if focus_score > 0.7 else "medium" if focus_score > 0.4 else "low"
        
        # Template Library
        templates = {
            ("high", "high"): [
                "Outstanding work! Your reading was accurate and you stayed completely focused.",
                "You're on fire today! Great pronunciation and steady attention."
            ],
            ("high", "medium"): [
                "Good reading! You nailed the words. Try to keep your eyes on the screen a bit more.",
                "Your accuracy is great. Let's work on staying steady next time."
            ],
            ("high", "low"): [
                "You read the words correctly, but you seem a bit distracted. Maybe take a quick stretch?",
                "Great accuracy, but I noticed you looking away. Let's try to focus for just 2 more minutes."
            ],
            ("medium", "high"): [
                "Good focus! There were a few tricky words, but you powered through.",
                "I like how attentive you are. Let's practice some of those harder sounds together."
            ],
            ("medium", "medium"): [
                "Solid effort. You're doing okay, just keep practicing those long words.",
                "Nice job. Remember to pause at periods to catch your breath."
            ],
            ("medium", "low"): [
                "It looks like you're getting tired. The words are getting a bit mixed up. Want a break?",
                "Let's pause. Focus is key to getting these words right."
            ],
            ("low", "high"): [
                "I admire your focus! This text was really hard, wasn't it? Let's try something easier.",
                "You stayed with it, which is great. Don't worry about the mistakes, we'll fix them."
            ],
            ("low", "medium"): [
                "That was a tough one. You stumbled a bit, but that's how we learn.",
                "Let's slow down. Read one word at a time."
            ],
            ("low", "low"): [
                "This seems too difficult right now and you look tired. Let's stop and play a game instead.",
                "I think we need a break. We can try this again later when you're fresh."
            ]
        }
        
        options = templates.get((accuracy_state, focus_state), ["Good effort today!"])
        feedback = random.choice(options)
        
        # Add specific stats
        feedback += f" (Accuracy: {int((1-error_rate)*100)}%, Focus: {int(focus_score*100)}%)"
        
        return feedback

class AdaptAgent:
    """Agent for Adaptive Difficulty using ML."""
    def __init__(self):
        self.model = load_model()

    def adapt(self, error_rate, wpm, focus_score):
        """Calculates the next difficulty index."""
        next_difficulty = predict_difficulty(self.model, error_rate, wpm, focus_score)
        return next_difficulty

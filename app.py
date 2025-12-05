import streamlit as st
import numpy as np
import cv2
import speech_recognition as sr
from agents import ListenAgent, ObserveAgent, AdaptAgent, AssistAgent, MentorAgent
import time

def main():
    st.set_page_config(page_title="Dyslexia Reading Assistant", page_icon="📖", layout="wide")
    
    st.title("📖 AI Reading Assistant")
    st.markdown("A dynamic, multisensory reading assistant powered by AI agents.")

    # Initialize Agents (st.cache_resource to avoid reloading heavy models if we had any)
    if 'listen_agent' not in st.session_state:
        st.session_state.listen_agent = ListenAgent()
        st.session_state.observe_agent = ObserveAgent()
        st.session_state.adapt_agent = AdaptAgent()
        st.session_state.assist_agent = AssistAgent()
        st.session_state.mentor_agent = MentorAgent()

        # Session State Variables
        st.session_state.current_difficulty = 0.5
        st.session_state.history = []

    # --- Sidebar ---
    st.sidebar.header("Session Stats")
    st.sidebar.metric("Current Difficulty", f"{st.session_state.current_difficulty:.2f}")
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.sidebar.text(f"Last Error Rate: {last['error']:.2f}")
        st.sidebar.text(f"Last Focus: {last['focus']:.2f}")

    # --- Main Content ---
    
    # 1. Reading Passage Display
    st.subheader("Reading Passage")
    
    # Text changes based on difficulty (Simulated content DB)
    passages = {
        "easy": "The sun is hot. The sky is blue. I like to play.",
        "medium": "The quick brown fox jumps over the lazy dog. It was a sunny day in the park.",
        "hard": "Photosynthesis is the process used by plants to convert light energy into chemical energy."
    }
    
    level = "medium"
    if st.session_state.current_difficulty < 0.3: level = "easy"
    elif st.session_state.current_difficulty > 0.7: level = "hard"
    
    target_text = passages[level]
    
    # Apply Visual Assistance
    assistance = st.session_state.assist_agent.provide_assistance(st.session_state.current_difficulty)
    
    style = ""
    if "font_spacing_wide" in assistance["visual_cues"]:
        style += "letter-spacing: 3px; line-height: 2.0; "
    if "syllable_segmentation" in assistance["visual_cues"]:
        # Naive syllable simulation for demo
        target_text = target_text.replace("e", "e-").replace("o", "o-").replace("a", "a-").replace("i", "i-").replace("u", "u-").replace("y", "y-")
    
    bg_color = "transparent"
    if assistance["highlight_color"] == "yellow": bg_color = "#fff9c4"
    elif assistance["highlight_color"] == "light_blue": bg_color = "#e1f5fe"

    st.markdown(f"""
    <div style="padding: 20px; background-color: {bg_color}; border-radius: 10px; font-size: 24px; {style}">
        {target_text}
    </div>
    """, unsafe_allow_html=True)
    
    if assistance["tts_enabled"]:
        st.caption("🔊 TTS Hints Enabled: Hover over hard words (Simulation)")

    st.divider()

    # 2. Input Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("👁️ Observe (Webcam)")
        camera_input = st.camera_input("Take a snapshot of your face while reading")
    
    with col2:
        st.header("👂 Listen (Microphone)")
        # Use the native audio input widget (Streamlit 1.40+)
        audio_input = st.audio_input("Record your voice")
        
        if audio_input is not None:
             st.success("Audio recorded!")

    if st.button("Analyze Session"):
        wpm = 0
        error_rate = 0.0
        focus_score = 0.5
        
        # --- Processing ---
        with st.spinner("Agents processing..."):
            
            # LISTEN
            if audio_input:
                # Pass the file-like object directly
                error_rate, wpm = st.session_state.listen_agent.listen_from_file(audio_input, target_text)
                if error_rate == 1.0 and wpm == 0:
                     st.warning("Could not understand audio. Please try again.")
                else:
                     st.success(f"Audio processed: {wpm} WPM, {int((1-error_rate)*100)}% Accuracy")
            else:
                # Fallback to mock behavior for demo/testing without mic
                # Only if they click analyze without recording, we assume simulation or just skip
                # Let's just simulate if no input to show the flow
                st.info("No audio recorded. Simulating...")
                error_rate = np.random.uniform(0.0, 0.2)
                wpm = np.random.randint(60, 150)
                
            
            # OBSERVE
            if camera_input:
                # Convert buffer to image for opencv
                file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                focus_score = st.session_state.observe_agent.analyze_image(image)
                st.success(f"Image processed: Focus Score {focus_score:.2f}")
            else:
                # No camera input, simulate
                focus_score = np.random.uniform(0.4, 1.0)
                st.info(f"No camera input. Simulated Focus Score {focus_score:.2f}")

            # ADAPT
            next_diff = st.session_state.adapt_agent.adapt(error_rate, wpm, focus_score)
            st.session_state.current_difficulty = next_diff
            
            # MENTOR
            feedback = st.session_state.mentor_agent.provide_feedback(error_rate, focus_score)
            
            # Display Results
            st.session_state.history.append({"error": error_rate, "focus": focus_score, "diff": next_diff})
            
            st.divider()
            st.header("🤖 Mentor Feedback")
            st.info(f"🗣️ {feedback}")
            # st.rerun() - Removed to keep feedback visible

if __name__ == "__main__":
    main()

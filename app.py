import streamlit as st
import tempfile
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import os

st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺")
st.title("🩺 AI Medical Voice Agent")

st.caption(
    "Speak your health question. This AI provides general medical information only — not diagnosis or treatment."
)

api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY missing in Streamlit secrets.")
    st.stop()

genai.configure(api_key=api_key)

def get_available_model():
    try:
        models = genai.list_models()
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                return m.name
        return None
    except Exception:
        return None

model_name = get_available_model()

if not model_name:
    st.error("No available Gemini models found.")
    st.stop()

st.markdown("---")

audio_bytes = st.audio_input("🎙 Speak your question")

if audio_bytes is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes.read())
        audio_path = tmpfile.name

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data)

        st.subheader("🗣 You said:")
        st.write(user_text)

    except Exception:
        st.error("Speech recognition failed. Please try again.")
        os.remove(audio_path)
        st.stop()

    os.remove(audio_path)

    emergency_keywords = [
        "chest pain",
        "suicidal",
        "suicide",
        "can't breathe",
        "difficulty breathing",
        "severe bleeding",
        "heart attack",
        "stroke"
    ]

    if any(word in user_text.lower() for word in emergency_keywords):
        st.warning(
            "⚠ If this may be a medical emergency, please seek immediate medical care."
        )

    prompt = (
        "You are a safe medical information assistant. "
        "Provide general, evidence-based health guidance only. "
        "Do not diagnose or prescribe treatments.\n\n"
        f"Patient question: {user_text}"
    )

    st.info("Generating response...")

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        if not response or not hasattr(response, "text"):
            st.error("Invalid response from AI.")
            st.stop()

        ai_text = response.text

    except Exception:
        st.error("Error generating response.")
        st.stop()

    st.subheader("🤖 AI Response:")
    st.write(ai_text)

    try:
        tts = gTTS(ai_text[:3000])
        tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tts_file.name)
        st.audio(tts_file.name)
        os.remove(tts_file.name)
    except Exception:
        st.warning("Speech synthesis failed, but text response is shown above.")

    st.markdown("---")
    st.markdown("### 🙏 Thank you for using AI Medical Voice Agent by Aditya Ranjan")

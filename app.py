import streamlit as st
import tempfile
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import os

api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Missing GOOGLE_API_KEY in Streamlit secrets.")
    st.stop()

genai.configure(api_key=api_key)

st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺")
st.title("🩺 AI Medical Voice Agent")
st.caption(
    "Speak your health question. "
    "This AI provides general medical information only — not diagnosis or treatment."
)

st.markdown("---")

audio_bytes = st.audio_input("🎙 Speak your question")

if audio_bytes is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes.read())
        audio_path = tmpfile.name

    st.success("✅ Audio received!")

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data)

        st.subheader("🗣 You said:")
        st.write(user_text)

    except sr.UnknownValueError:
        st.error("⚠ Could not understand audio. Please try again.")
        os.remove(audio_path)
        st.stop()

    except sr.RequestError:
        st.error("⚠ Speech recognition service unavailable.")
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
            "⚠ If this may be a medical emergency, "
            "please seek immediate medical care or call emergency services."
        )

    st.info("💬 Generating response...")

    prompt = (
        "You are a factual and safe medical information assistant. "
        "Provide helpful, general, evidence-based health guidance. "
        "Do not diagnose, prescribe medication, or replace a doctor. "
        "Encourage seeking professional care when appropriate.\n\n"
        f"Patient question: {user_text}"
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        if not response or not hasattr(response, "text"):
            st.error("⚠ Invalid response from Gemini.")
            st.stop()

        ai_text = response.text

    except Exception:
        st.error("⚠ Error connecting to Gemini API.")
        st.stop()

    st.subheader("🤖 AI Response:")
    st.write(ai_text)

    try:
        tts = gTTS(ai_text[:3000])
        tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tts_file.name)

        st.audio(tts_file.name, format="audio/mp3")

        os.remove(tts_file.name)

        st.success("🎯 Response generated successfully!")

    except Exception:
        st.warning("Speech synthesis failed, but text response is available above.")

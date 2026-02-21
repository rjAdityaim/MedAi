import streamlit as st
import tempfile
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import os

st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺")

st.title("🩺 AI Medical Voice Agent (Debug Mode)")

api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY missing in Streamlit secrets.")
    st.stop()

st.write("API Key Loaded:", api_key[:10] + "...")

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Gemini configuration error: {e}")
    st.stop()

st.subheader("🔎 Testing Gemini Connection")

try:
    test_model = genai.GenerativeModel("gemini-1.5-flash")
    test_response = test_model.generate_content("Say hello")
    st.success("Gemini test successful")
    st.write("Test response:", test_response.text)
except Exception as e:
    st.error("Gemini test failed")
    st.error(str(e))
    st.stop()

st.markdown("---")

audio_bytes = st.audio_input("🎙 Speak your question")

if audio_bytes is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio_bytes.read())
        audio_path = tmpfile.name

    st.success("Audio received")

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data)

        st.subheader("You said:")
        st.write(user_text)

    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        os.remove(audio_path)
        st.stop()

    os.remove(audio_path)

    prompt = (
        "You are a safe medical information assistant. "
        "Provide general health guidance only.\n\n"
        f"Patient question: {user_text}"
    )

    st.info("Generating Gemini response...")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        if not response:
            st.error("Empty response from Gemini")
            st.stop()

        ai_text = response.text
        st.subheader("AI Response:")
        st.write(ai_text)

    except Exception as e:
        st.error("Gemini generation error:")
        st.error(str(e))
        st.stop()

    try:
        tts = gTTS(ai_text[:3000])
        tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tts_file.name)
        st.audio(tts_file.name)
        os.remove(tts_file.name)
    except Exception as e:
        st.warning(f"TTS error: {e}")

import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Initialize models
emotion_detector = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Session state for conversation history and mood tracking
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "mood_data" not in st.session_state:
    st.session_state.mood_data = []

def detect_emotion(text):
    result = emotion_detector(text)[0]
    return result["label"], result["score"]

def query_local_model(prompt):
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def log_mood(mood_score):
    st.session_state.mood_data.append(mood_score)
    st.success("Mood logged successfully!")

def show_mood_graph():
    if st.session_state.mood_data:
        df = pd.DataFrame({"Day": range(1, len(st.session_state.mood_data) + 1), "Mood Score": st.session_state.mood_data})
        fig = px.line(df, x="Day", y="Mood Score", title="Your Mood Journey 🌱", labels={"Mood Score": "Score (1-5)"})
        st.plotly_chart(fig)
    else:
        st.warning("No mood data available yet. Log your mood to see trends!")

def main():
    st.title("AI Mental Health Companion 🌟")
    st.write("Share your thoughts—I'm listening.")

    # Input type selection
    input_type = st.radio("How would you like to share?", ("Text", "Voice"))
    user_input = ""

    if input_type == "Text":
        user_input = st.text_input("Type here:")
    else:
        if st.button("Start Recording"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                try:
                    user_input = recognizer.recognize_google(audio)
                    st.write(f"You said: {user_input}")
                except:
                    st.error("Audio not recognized. Try again.")

    # Process input
    if user_input:
        # Detect emotion
        emotion, confidence = detect_emotion(user_input)
        st.write(f"Detected emotion: **{emotion}** (confidence: {confidence:.2f})")

        # Generate response using local model
        prompt = f"""
        You are a mental health companion. The user is feeling **{emotion}**. 
        Their message: "{user_input}"
        Conversation history: {st.session_state.conversation_history}
        Respond with empathy, avoid clichés, and suggest actionable steps.
        """
        bot_response = query_local_model(prompt)
        st.write(f"**Bot**: {bot_response}")

        # Update conversation history
        st.session_state.conversation_history.append(
            {"user": user_input, "bot": bot_response}
        )

        # Crisis escalation (if extreme emotions)
        if emotion in ["sadness", "fear"] and confidence > 0.8:
            st.warning("You seem deeply upset. Would you like me to suggest professional resources?")
            if st.button("Yes, show resources"):
                st.write("📞 **Hotlines**: 988 (US Suicide Hotline)")
                st.write("🌐 **Online Therapy**: [BetterHelp](https://www.betterhelp.com)")

    # Gamified mood tracking
    st.header("Mood Tracker 📊")
    mood_score = st.slider("Rate your mood today (1-5):", 1, 5)
    if st.button("Log Mood"):
        log_mood(mood_score)
    show_mood_graph()

if __name__ == "__main__":
    main()
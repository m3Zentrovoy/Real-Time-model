import streamlit as st
import numpy as np
import pandas as pd
import librosa
import emotion_ai_engine as processor

st.set_page_config(page_title="Audio File Validator", layout="wide")
st.title("📂 Model Validation on Files (Wav2Vec2)")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Audio (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    try:
        st.info("Loading and resampling to 16000 Hz... (This may take a while)")
        
        # Librosa loads and immediately resamples to 16kHz (standard for Wav2Vec)
        audio_data, sr = librosa.load(uploaded_file, sr=16000)
        
        st.audio(uploaded_file, format='audio/wav')
        st.write(f"Duration: {len(audio_data)/sr:.2f} sec")
        
        # --- 2. RUN ANALYSIS ---
        if st.button("Run Analysis"):
            # Settings: split into 1-second chunks
            chunk_duration = 1.0 
            step = int(chunk_duration * sr)
            
            results = []
            progress_bar = st.progress(0)
            
            # Iterate through file with window
            for i in range(0, len(audio_data), step):
                chunk = audio_data[i : i + step]
                
                # Ignore short residuals (< 0.5 sec) at the end of file
                if len(chunk) < step * 0.5:
                    continue
                
                # CALL NEURAL NETWORK
                scores = processor.get_all_scores(chunk)
                
                if scores:
                    timestamp = i / sr
                    row = {"Time": timestamp}
                    row.update(scores) # Add probabilities of all emotions
                    results.append(row)
                
                # Update progress bar
                if len(audio_data) > 0:
                    progress_bar.progress(min(i / len(audio_data), 1.0))
            
            progress_bar.empty()
            
            # --- 3. VISUALIZATION ---
            if results:
                df = pd.DataFrame(results)
                df = df.set_index("Time")
                
                # Probability chart
                st.subheader("Emotion Dynamics (Probabilities)")
                st.line_chart(df[['neutral', 'happy', 'angry', 'sad', 'fear', 'surprise']])
                
                # Data table
                st.subheader("Detailed Table")
                st.dataframe(df.style.highlight_max(axis=1))
            else:
                st.warning("No data for analysis (file might be empty).")

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.write("Ensure librosa is installed: pip install librosa")
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Load model and reference averages
model = joblib.load('amp_model.pkl')
class_averages = joblib.load('class_averages.pkl')

st.title("Antimicrobial Peptide (AMP) Predictor")
st.write("Paste a peptide sequence to predict antimicrobial activity and compare its properties.")

sequence = st.text_input("Peptide Sequence (amino acid letters only):").upper().strip()

if sequence:
    try:
        analysis = ProteinAnalysis(sequence)
        features = {
            'length': len(sequence),
            'molecular_weight': analysis.molecular_weight(),
            'aromaticity': analysis.aromaticity(),
            'instability_index': analysis.instability_index(),
            'isoelectric_point': analysis.isoelectric_point(),
            'gravy': analysis.gravy(),
            'charge_at_pH7': analysis.charge_at_pH(7.0),
        }
        input_df = pd.DataFrame([features])

        # Prediction
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        label = "Antimicrobial (AMP)" if pred == 1 else "Non-antimicrobial"
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {proba:.2%}")

        import numpy as np

        # Normalize for visualization
        compare_df = pd.concat([class_averages, input_df.rename(index={0: 'Your Sequence'})])
        normalized_df = (compare_df - compare_df.min()) / (compare_df.max() - compare_df.min())

        st.subheader("Feature Comparison (normalized)")

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(normalized_df.columns))
        width = 0.25
        colors = ['#6B8CAE', '#C97B4A', '#5A9367']  # muted blue, muted orange, muted green


        for i, row_name in enumerate(normalized_df.index):
                ax.bar(x + i*width, normalized_df.loc[row_name], width, label=row_name, color=colors[i])

        ax.set_xticks(x + width)
        ax.set_xticklabels(normalized_df.columns, rotation=45, ha='right')
        ax.legend()
        ax.set_ylabel("Normalized value (0-1)")
        st.pyplot(fig)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader(f"Prediction: {label}")
            st.metric("Confidence", f"{proba:.1%}")

        with col2:
            st.subheader("Feature Comparison")
            st.pyplot(fig)
        with st.expander("See raw feature values"):
            st.dataframe(compare_df)
    except Exception:
        st.error("Invalid sequence — use only standard amino acid letters (A-Z, no numbers/symbols).")

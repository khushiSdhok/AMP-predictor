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

               # Normalize for visualization (min-max scale each feature 0-1)
        compare_df = pd.concat([class_averages, input_df.rename(index={0: 'Your Sequence'})])
        normalized_df = (compare_df - compare_df.min()) / (compare_df.max() - compare_df.min())

        st.subheader("Feature Comparison (normalized)")
        st.bar_chart(normalized_df.T)

        with st.expander("See raw feature values"):
            st.dataframe(compare_df)
    except Exception:
        st.error("Invalid sequence — use only standard amino acid letters (A-Z, no numbers/symbols).")

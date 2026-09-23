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

        # Comparison chart: your sequence vs class averages
        compare_df = pd.concat([class_averages, input_df.rename(index={0: 'Your Sequence'})])
        st.subheader("Feature Comparison")
        st.bar_chart(compare_df.T)

    except Exception:
        st.error("Invalid sequence — use only standard amino acid letters (A-Z, no numbers/symbols).")
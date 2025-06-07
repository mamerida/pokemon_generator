# app.py
import streamlit as st

st.title("Mi primera app con Streamlit 🎈")

nombre = st.text_input("¿Cómo te llamas?")
if nombre:
    st.write(f"¡Hola, {nombre}!")

numero = st.slider("Elige un número", 1, 100)
st.write(f"El número que elegiste es: {numero}")
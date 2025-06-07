import streamlit as st
from components.pokemon_gallery import mostrar_galeria_pokemon_paginada 
import os
import pandas as pd



def main():
    st.set_page_config(page_title="Generador de Pokémon Aleatorios", page_icon="🧬")
    df = pd.read_csv("data/pokemon.csv")
    rows = df.to_dict(orient="records")

    # Título principal con ícono
    st.title("🧬 Generador de Pokémon Aleatorios")

    # Descripción persuasiva con "automágicamente" destacado
    st.markdown(
        """
        Deja volar tu imaginación generando todos los Pokémon que quieras.  
        Selecciona uno de nuestra lista y generaremos uno <span style="font-size: 22px;"><b>automágicamente ✨</b></span>  
        Descubrí nuevas combinaciones, explorá criaturas que jamás imaginaste y compartí tus creaciones con amigos.  
        Esta app es perfecta para fans, jugadores y entrenadores que buscan inspiración, diversión o nuevos retos.
        """,
        unsafe_allow_html=True  # Permite usar HTML
    )

    st.subheader("Selecciona tu Pokémon favorito")
    folder = "data/pokemon/"
    images = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 0
    if "selected_pokemon" not in st.session_state:
        st.session_state["selected_pokemon"] = []
    lang_dict = {
        "select": "Seleccionar",
        "previous": "<--",
        "next": "-->",
        "pagination": "Página {current} de {total}",
        "selected": "Seleccionados",
    }

    mostrar_galeria_pokemon_paginada(
        images=images,
        page_number=st.session_state["page_number"],
        images_per_page=8,
        images_per_row=4,
        pokemon_rows=rows,
        lang_dict=lang_dict,
    )

    # Botón para generar
    if st.button("✨ Generar Pokémon ✨"):
        if pokemon_seleccionado:
            st.success(f"¡Tu Pokémon generado a partir de {pokemon_seleccionado} está listo!")
        else:
            st.warning("Por favor, seleccioná un Pokémon para comenzar.")

if __name__ == "__main__":
    main()

# streamlit run app.py
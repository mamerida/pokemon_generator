import streamlit as st
from components.pokemon_gallery import mostrar_galeria_pokemon_paginada, mostrar_tipos_pokemon 
import os
import pandas as pd
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from prod.models.utils import (
    load_dataset,
    load_model,
    generate_by_neighbors_image,
    buscar_imagen_por_nombre,
    interpolar_pokemon_por_nombre,
    generate_representative_pokemon,
    generate_multiple_by_neighbors
)

st.set_page_config(page_title="Generador de Pokémon Aleatorios", page_icon="🧬")

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_dataset():
    return load_dataset()

@st.cache_data
def get_tipos(_dataset):
    return sorted(list(set([str(label) for _, label, _ in _dataset])))

def mostrar_grilla_pokemon():
    df = pd.read_csv("data/pokemon.csv")
    rows = df.to_dict(orient="records")
    # Buscador
    busqueda = st.text_input("🔍 Buscar Pokémon por nombre")
    if st.button("❌ Quitar selección"):
        st.session_state["selected_pokemon"] = []

    folder = "data/pokemon/"
    images = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and busqueda.lower() in f.lower()
    ])

    # Mostrar grilla de Pokémon
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
        modo=st.session_state["modo_generacion"] 
    )

def main():
    dataset = get_dataset()
    model = get_model()
    tipos = get_tipos(dataset)

    df = pd.read_csv("data/pokemon.csv")
    rows = df.to_dict(orient="records")

    st.title("🧬 Generador de Pokémon Aleatorios")

    # Inicialización de estados
    if "modo_generacion" not in st.session_state:
        st.session_state["modo_generacion"] = None
    if "selected_pokemon" not in st.session_state:
        st.session_state["selected_pokemon"] = []
    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 0

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Descripciones
        - <span style="font-size: 27px;"><b>Generar Pokémon: </b></span>Selecciona uno de nuestra lista, elegi la cantidad de pokémon a combinar y generaremos uno <span style="font-size: 22px;"><b>automágicamente ✨ (recomendamos menos de 5 pokémon para combinar)</b></span> </br></br>
        - <span style="font-size: 27px;"><b>Generar Multi Pokémon: </b></span>Selecciona uno de nuestra lista, elegi la cantidad de pokémon que debe tener en cuenta para crear y generaremos 4 Pokémon <span style="font-size: 22px;"><b>nuevos ✨ (recomendamos mas de 5 pokémon a tener en cuenta)</b></span> </br></br>
        - <span style="font-size: 27px;"><b>Interpolar Pokémon: </b></span>Selecciona dos de nuestra lista, elegi la cantidad de pasos y te mostraremos como los <span style="font-size: 22px;"><b>combinamos 🔀</b></span> de a poco </br></br>
        - <span style="font-size: 27px;"><b>Generar representativo: </b></span>Selecciona un tipo de nuestra lista y te generaremos el <span style="font-size: 22px;"><b>Pokémon representativo del tipo elegido 🎉​​</b></span> </br>
        
        """,
        unsafe_allow_html=True)  # Permite usar HTML

    with col2:

        st.markdown("### Acciones")

        botones = {
            "✨ Generar Pokémon ✨": "generar",
            "🎨 Generar Múltiple 🎨": "multiple",
            "🔀 Interpolar 🔀": "interpolar",
            "🏆 Representativo por tipo 🏆": "tipo"
        }

        for label, modo in botones.items():
            if st.button(label):
                if st.session_state.get("modo_generacion") != modo:
                    st.session_state["modo_generacion"] = modo
                    st.session_state["selected_pokemon"] = []  # limpiar selección de Pokémon
                    st.session_state["tipo_seleccionado"] = None  # limpiar selección de tipo
                    st.session_state["page_number"] = 0  # opcional: resetear paginación
                    st.rerun()

    st.markdown("""
        Descubrí nuevas combinaciones, explorá criaturas que jamás imaginaste y compartí tus creaciones con amigos.  
        Esta app es perfecta para fans, jugadores y entrenadores que buscan inspiración, diversión o nuevos retos.
        """,
        unsafe_allow_html=True)  # Permite usar HTML

    modo = st.session_state.get("modo_generacion")
    seleccionados = st.session_state.get("selected_pokemon", [])
    tipo = st.session_state.get("tipo_seleccionado", None)

    if modo in ["generar", "multiple", "interpolar"]:
        mostrar_grilla_pokemon()

    # Mostrar controles dinámicos según el modo
    if modo == "generar":
        k = st.number_input("Vecinos a combinar", 1, 10, 3)
        if st.button("🚀 Generar Pokémon"):
            img, vecinos = generate_by_neighbors_image(model, dataset, seleccionados[0], k=k)
            st.image(to_pil_image(img), caption="Pokémon generado", use_container_width=True)
            st.markdown("### 🧬 Vecinos utilizados:")
            cols = st.columns(len(vecinos))
            for i, name in enumerate(vecinos):
                with cols[i]:
                    ruta = buscar_imagen_por_nombre(name)
                    if ruta: st.image(Image.open(ruta), caption=name, use_container_width=True)

    elif modo == "multiple":
        k = st.number_input("Vecinos a considerar", 1, 10, 3)
        if st.button("🧪 Generar Múltiples"):
            imgs, vecinos = generate_multiple_by_neighbors(model, dataset, seleccionados[0], k=k)
            st.markdown("### Nuevos Pokémon")
            cols = st.columns(len(imgs))
            for i, img in enumerate(imgs):
                with cols[i]: st.image(img, caption=f"#{i+1}", use_container_width=True)

            st.markdown("### Vecinos usados:")
            cols = st.columns(len(vecinos))
            for i, name in enumerate(vecinos):
                with cols[i]:
                    ruta = buscar_imagen_por_nombre(name)
                    if ruta: st.image(Image.open(ruta), caption=name, use_container_width=True)

    elif modo == "interpolar":
        steps = st.number_input("Pasos de interpolación", 2, 30, 10)
        if len(seleccionados) == 2:
            if st.button("🔀 Interpolar ahora"):
                imgs = interpolar_pokemon_por_nombre(model, dataset, seleccionados[0], seleccionados[1], steps)
                st.markdown("### Transición:")
                for i in range(0, len(imgs), 5):
                    cols = st.columns(min(5, len(imgs) - i))
                    for j, img in enumerate(imgs[i:i+5]):
                        with cols[j]: st.image(to_pil_image(img), width=150, caption=f"Paso {i+j+1}")
        elif len(seleccionados) == 1:
            st.info("Seleccioná un segundo Pokémon para interpolar.")

    elif modo == "tipo":
        lang_dict = {
            "select_type": "Selecciona un tipo de Pokémon",
            "type_selected": "Tipo seleccionado"
        }
        mostrar_tipos_pokemon(dataset, lang_dict, tipos)
        tipo = st.session_state.get("tipo_seleccionado")
        if tipo and st.button("✨ Generar Pokémon representativo"):
            img = generate_representative_pokemon(model, dataset, tipo)
            if img:
                st.image(img, caption=f"Representativo del tipo {tipo}", use_container_width=True)
            else:
                st.warning("No se pudo generar el Pokémon.")

    elif len(seleccionados) > 2:
        st.warning("Solo se pueden seleccionar hasta 2 Pokémon.")

if __name__ == "__main__":
    main()

# streamlit run app.py
# python -m streamlit run app.py
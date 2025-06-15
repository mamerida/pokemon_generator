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

def main():
    dataset = get_dataset()
    model = get_model()
    tipos = get_tipos(dataset)

    df = pd.read_csv("data/pokemon.csv")
    rows = df.to_dict(orient="records")

    # Título principal con ícono
    st.title("🧬 Generador de Pokémon Aleatorios")

    # Descripción persuasiva con "automágicamente" destacado
    st.markdown(
        """
        Deja volar tu imaginación generando todos los Pokémon que quieras.  </br></br>
        <span style="font-size: 27px;"><b>Generar Pokémon: </b></span>Selecciona uno de nuestra lista, elegi la cantidad de pokémon a combinar y generaremos uno <span style="font-size: 22px;"><b>automágicamente ✨</b></span> </br></br>
        <span style="font-size: 27px;"><b>Generar Multi Pokémon: </b></span>Selecciona uno de nuestra lista, elegi la cantidad de pokémon que debe tener en cuenta para crear y generaremos 4 Pokémon <span style="font-size: 22px;"><b>nuevos ✨</b></span> </br></br>
        <span style="font-size: 27px;"><b>Interpolar Pokémon: </b></span>Selecciona dos de nuestra lista y te mostraremos como los <span style="font-size: 22px;"><b>combinamos 🔀</b></span> de a poco </br></br>
        <span style="font-size: 27px;"><b>Generar representativo: </b></span>Selecciona un tipo de nuestra lista y te generaremos el <span style="font-size: 22px;"><b>Pokémon representativo del tipo elegido 🎉​​</b></span> </br>
        
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

    lang_dict = {
        "select_type": "Selecciona un tipo de Pokémon",
        "type_selected": "Tipo seleccionado"
    }

    mostrar_tipos_pokemon(dataset, lang_dict, tipos)

    # Obtener selección actual
    seleccionados = st.session_state.get("selected_pokemon", [])
    seleccion_tipo = st.session_state.get("tipo_seleccionado")

    # Cambiar de tipo a Pokémon
    if len(seleccionados) in [1, 2] and st.session_state.get("modo_actual") != "pokemon":
        st.session_state["tipo_seleccionado"] = None
        st.session_state["modo_actual"] = "pokemon"
        st.rerun()

    # Cambiar de Pokémon a tipo
    elif seleccion_tipo and st.session_state.get("modo_actual") != "tipo":
        st.session_state["selected_pokemon"] = []
        st.session_state["modo_actual"] = "tipo"
        st.rerun()

    # Mostrar botones según cantidad seleccionada
    if len(seleccionados) == 1:
        k = st.number_input("Cantidad de Pokémon a combinar", min_value=1, max_value=10, value=2, step=1)
        if st.button("✨ Generar Pokémon ✨"):
            nombre = seleccionados[0]
            gen_img, neighbor_names = generate_by_neighbors_image(
                model, dataset, nombre, device=None, k=int(k)
            )
            st.success(f"¡Tu Pokémon generado a partir de {nombre} está listo!")
            st.image(to_pil_image(gen_img), caption="Pokémon generado", use_container_width=True)

            st.markdown("### 🧬 Vecinos utilizados para generar el Pokémon:")
            cols = st.columns(len(neighbor_names))
            for i, name in enumerate(neighbor_names):
                with cols[i]:
                    ruta_imagen = buscar_imagen_por_nombre(name)
                    if ruta_imagen and os.path.exists(ruta_imagen):
                        st.image(Image.open(ruta_imagen), caption=name, use_container_width=True)
                    else:
                        st.warning(f"No se encontró imagen para: {name}")

        if st.button("✨ Generar Multi Pokémon ✨"):
            nombre = seleccionados[0]
            generated_imgs, neighbor_names = generate_multiple_by_neighbors(model, dataset, nombre, k=int(k), num_outputs=4)
            st.markdown("### 🎨 Pokémon generados:")
            cols = st.columns(len(generated_imgs))
            for i, img in enumerate(generated_imgs):
                with cols[i]:
                    st.image(img, caption=f"Generado #{i+1}", use_container_width=True)

            st.markdown("### 🧬 Vecinos utilizados para generar los Pokémon:")
            cols = st.columns(len(neighbor_names))
            for i, name in enumerate(neighbor_names):
                with cols[i]:
                    ruta_imagen = buscar_imagen_por_nombre(name)
                    if ruta_imagen and os.path.exists(ruta_imagen):
                        st.image(Image.open(ruta_imagen), caption=name, use_container_width=True)
                    else:
                        st.warning(f"No se encontró imagen para: {name}")

    elif len(seleccionados) == 2:
        st.info("Se seleccionaron dos Pokémon. Podés interpolarlos.")
        if st.button("🔀 Interpolar Pokémon"):
            st.success(f"¡Listo para interpolar entre {seleccionados[0]} y {seleccionados[1]}!")
            interpolados = interpolar_pokemon_por_nombre(model, dataset, seleccionados[0], seleccionados[1], steps=20)

            st.markdown("### 🔀 Interpolación entre Pokémon seleccionados")
            for i in range(0, len(interpolados), 5):
                cols = st.columns(min(5, len(interpolados) - i))
                for j, img_tensor in enumerate(interpolados[i:i+5]):
                    with cols[j]:
                        st.image(to_pil_image(img_tensor), width=150, caption=f"Paso {i + j + 1}")

    elif seleccion_tipo:
        st.info("Se seleccionó un tipo de Pokémon. Podés crear uno nuevo de ese tipo.")
        if st.button("🎉 Generar Pokémon representativo 🎉"):
            st.success(f"¡Listo para generar un Pokémon de tipo {seleccion_tipo}!")
            img_supremo = generate_representative_pokemon(model, dataset, seleccion_tipo)
            if img_supremo:
                st.image(img_supremo, caption=f"Pokémon del tipo {seleccion_tipo}", use_container_width=True)
            else:
                st.warning("No se pudieron generar imágenes para ese tipo.")
    else:
        if len(seleccionados) > 2:
            st.warning("Por favor, seleccioná solo 1 o 2 Pokémon.")

if __name__ == "__main__":
    main()

# streamlit run app.py
# python -m streamlit run app.py
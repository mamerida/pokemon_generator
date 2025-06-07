import streamlit as st
import os

def mostrar_galeria_pokemon_paginada(images, page_number, images_per_page, images_per_row, pokemon_rows, lang_dict):
    total_pages = (len(images) - 1) // images_per_page
    start_idx = page_number * images_per_page
    end_idx = min(start_idx + images_per_page, len(images))
    images_to_show = images[start_idx:end_idx]

    # Inicializar control si no existe
    if "pokemon_selected" not in st.session_state:
        st.session_state["pokemon_selected"] = False

    with st.expander("Galería Pokémon (desplázate hacia abajo)", expanded=True):
        for i in range(0, len(images_to_show), images_per_row):
            cols = st.columns(images_per_row, gap="small")
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(images_to_show):
                    break
                img_path = images_to_show[idx]
                name = os.path.splitext(os.path.basename(img_path))[0]
                with col:
                    st.image(img_path, use_container_width=True)
                    st.write(name)
                    if st.button(lang_dict["select"], key=f"select_{start_idx + idx}"):
                        index = next((i for i, row in enumerate(pokemon_rows) if row["Name"] == name), -1)
                        st.session_state["selected_pokemon"] = [name]
                        st.session_state["pokemon_selected"] = True
                        st.rerun()

    st.write(lang_dict["pagination"].format(current=page_number + 1, total=total_pages + 1))

    colA, _, colB = st.columns([0.1, 0.8, 0.1], gap="small")
    with colA:
        if st.button(lang_dict["previous"], key="prev_page") and not st.session_state.get("pokemon_selected", False):
            if page_number > 0:
                st.session_state["page_number"] = page_number - 1
                st.rerun()
    with colB:
        if st.button(lang_dict["next"], key="next_page") and not st.session_state.get("pokemon_selected", False):
            if page_number < total_pages:
                st.session_state["page_number"] = page_number + 1
                st.rerun()

    if "selected_pokemon" in st.session_state and st.session_state["selected_pokemon"]:
        st.success(f"{lang_dict['selected']}: {', '.join(st.session_state['selected_pokemon'])}")

    # Resetear el flag después de mostrar todo
    st.session_state["pokemon_selected"] = False



import streamlit as st
import os

def mostrar_galeria_pokemon_paginada(images, page_number, images_per_page, images_per_row, pokemon_rows, lang_dict):
    total_pages = (len(images) - 1) // images_per_page
    start_idx = page_number * images_per_page
    end_idx = min(start_idx + images_per_page, len(images))
    images_to_show = images[start_idx:end_idx]

    if "selected_pokemon" not in st.session_state:
        st.session_state["selected_pokemon"] = []

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

                    seleccionados = st.session_state["selected_pokemon"]
                    is_selected = name in seleccionados

                    boton_texto = "❌ Quitar" if is_selected else lang_dict["select"]

                    if st.button(boton_texto, key=f"select_{start_idx + idx}"):
                        if is_selected:
                            seleccionados.remove(name)
                        else:
                            if len(seleccionados) < 2:
                                seleccionados.append(name)
                            else:
                                st.warning("Solo podés seleccionar hasta 2 Pokémon.")
                        st.session_state["selected_pokemon"] = seleccionados
                        st.rerun()

    st.write(lang_dict["pagination"].format(current=page_number + 1, total=total_pages + 1))

    colA, _, colB = st.columns([0.1, 0.8, 0.1], gap="small")
    with colA:
        if st.button(lang_dict["previous"], key="prev_page"):
            if page_number > 0:
                st.session_state["page_number"] = page_number - 1
                st.rerun()
    with colB:
        if st.button(lang_dict["next"], key="next_page"):
            if page_number < total_pages:
                st.session_state["page_number"] = page_number + 1
                st.rerun()

    seleccionados = st.session_state["selected_pokemon"]
    if seleccionados:
        st.success(f"{lang_dict['selected']}: {', '.join(seleccionados)}")


def mostrar_tipos_pokemon(dataset, lang_dict, tipos, columnas_por_fila=4):
    # Obtener tipos únicos
    # tipos = sorted(list(set([label for _, label, _ in dataset])))
    # tipos = sorted(list(set([str(label) for _, label, _ in dataset])))

    # Inicializar estado
    if "tipo_seleccionado" not in st.session_state:
        st.session_state["tipo_seleccionado"] = None

    st.markdown("### 🔎 " + lang_dict.get("select_type", "Selecciona un tipo de Pokémon"))

    for i in range(0, len(tipos), columnas_por_fila):
        cols = st.columns(min(columnas_por_fila, len(tipos) - i))
        for j, tipo in enumerate(tipos[i:i+columnas_por_fila]):
            with cols[j]:
                if st.button(tipo, key=f"tipo_{tipo}"):
                    st.session_state["tipo_seleccionado"] = tipo
                    st.rerun()

    # Mostrar selección
    if st.session_state["tipo_seleccionado"]:
        st.success(f"{lang_dict.get('type_selected', 'Tipo seleccionado')}: {st.session_state['tipo_seleccionado']}")
    

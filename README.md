# Pokémon Generator
![image](https://github.com/user-attachments/assets/0769e156-153e-45ad-ac5e-6d19d0776a4b)

Generador automático de imágenes de Pokémon mediante un modelo **Variational Autoencoder (VAE)** convolucional implementado en PyTorch, entrenado con imágenes y tipos de Pokémon.  
Este proyecto permite generar, explorar e interpolar nuevas imágenes de Pokémon directamente desde el espacio latente aprendido.

---

## 🚀 Descripción

Este repositorio implementa un **VAE convolucional** que aprende una representación latente compacta y continua para imágenes de Pokémon. El modelo se entrena con un dataset de imágenes y tipos, utilizando transformaciones y normalizaciones para mejorar la calidad de las reconstrucciones y generación.

Las principales funcionalidades son:

- Generación automática de nuevas imágenes de Pokémon desde el espacio latente.
- Interpolación entre imágenes para crear combinaciones visuales novedosas.
- Visualización y análisis de la estructura del espacio latente.

---

## 🧑‍🤝‍🧑 Colaboradores

Este proyecto fue desarrollado por:

- Mario Merida  
- Karen Ruiz  
- Sebastián Hofer  

---

## 🌐 Demo online

El modelo está desplegado automáticamente en la siguiente aplicación web interactiva:

👉 [https://pokemongenerator-equipo-6.streamlit.app/](https://pokemongenerator-equipo-6.streamlit.app/)

---

## 🔧 Tecnologías y librerías

- Python 3.8+  
- PyTorch  
- torchvision  
- matplotlib  
- scikit-learn (para t-SNE)  
- numpy  

---

## 📁 Estructura del repositorio
```bash
pokemon_generator/
│
├── data/ # Dataset preprocesado
├── models/ # Código del modelo VAE y arquitecturas
├── notebooks/ # Notebooks con experimentos y visualizaciones
├── scripts/ # Scripts para entrenamiento, generación y evaluación
├── outputs/ # Imágenes generadas y visualizaciones
├── requirements.txt # Dependencias
└── README.md
```

---

## 🏃‍♂️ Cómo usar

### Recomendación: usar pipenv para manejar dependencias y entorno virtual

Instalá pipenv siguiendo la documentación oficial:  
[https://pipenv.pypa.io/en/latest/](https://pipenv.pypa.io/en/latest/)

Luego, para instalar las dependencias y activar el entorno:

```bash
pipenv install -r requirements.txt
pipenv shell
```

```bash
 streamlit run app.py 
```


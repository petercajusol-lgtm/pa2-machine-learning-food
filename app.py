import streamlit as st
import pandas as pd
import joblib
import os

# Configuración de la página al estilo premium/minimalista
st.set_page_config(
    page_title="Predicciones Alimenticias",
    page_icon="🍲",
    layout="centered"
)

# --- REQUISITO OBLIGATORIO: DATOS DEL ESTUDIANTE ---
st.title("🍲 Predicción de Éxito de Ventas Alimenticias")
st.markdown("### **Estudiante:** Peter Alonso Cajusol Ramos")
st.markdown("### **Código ISIL:NRC:30506") #
st.markdown("[🔗 Ver Cuaderno de Google Colab (Modo Lector)](https://colab.research.google.com/drive/1EyjM3z9-vHslI96QfXFmZ27x-N_Bsemj?usp=sharing)")
st.markdown("---")

# Cargar el modelo entrenado y las columnas de referencia
@st.cache_resource
def load_model_assets():
    model = joblib.load("modelos/modelo_rf.pkl")
    model_columns = joblib.load("modelos/columnas_modelo.pkl")
    return model, model_columns

try:
    model, model_columns = load_model_assets()
    
    st.subheader("Ingrese los datos del producto alimenticio")
    
    # Formulario interactivo ordenado para el usuario
    col1, col2 = st.columns(2)
    
    with col1:
        calories = st.number_input("Calorías (kcal)", min_value=0.0, max_value=2000.0, value=250.0, step=10.0)
        protein_g = st.number_input("Proteínas (g)", min_value=0.0, max_value=200.0, value=15.0, step=1.0)
        total_fat_g = st.number_input("Grasas Totales (g)", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
        sodium_mg = st.number_input("Sodio (mg)", min_value=0.0, max_value=5000.0, value=300.0, step=50.0)
    
    with col2:
        total_carbs_g = st.number_input("Carbohidratos Totales (g)", min_value=0.0, max_value=500.0, value=30.0, step=5.0)
        sugars_g = st.number_input("Azúcares (g)", min_value=0.0, max_value=200.0, value=5.0, step=1.0)
        price_usd_normalized = st.number_input("Precio Normalizado (USD)", min_value=0.0, max_value=100.0, value=12.5, step=0.5)
        avg_rating = st.slider("Calificación Promedio (Rating)", min_value=1.0, max_value=5.0, value=4.2, step=0.1)

    # Inputs para variables categóricas (mapeadas a las columnas dummificadas)
    st.markdown("#### Selección de Categorías")
    brand_tier = st.selectbox("Gama de la Marca (Brand Tier)", ["Mid", "Budget", "Premium"])
    
    # Al hacer clic en el botón, construimos el dataframe espejo para predecir
    if st.button("Evaluar Producto"):
        # Estructurar los datos de entrada rellenando con ceros de acuerdo a las columnas del entrenamiento
        input_data = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Asignar variables numéricas directas
        input_data['calories'] = calories
        input_data['protein_g'] = protein_g
        input_data['total_fat_g'] = total_fat_g
        input_data['sodium_mg'] = sodium_mg
        input_data['total_carbs_g'] = total_carbs_g
        input_data['sugars_g'] = sugars_g
        input_data['price_usd_normalized'] = price_usd_normalized
        input_data['avg_rating'] = avg_rating
        
        # Asignar variables dummificadas si no corresponden al caso base (drop_first)
        if brand_tier == "Budget" and 'brand_tier_Budget' in model_columns:
            input_data['brand_tier_Budget'] = 1
        elif brand_tier == "Premium" and 'brand_tier_Premium' in model_columns:
            input_data['brand_tier_Premium'] = 1
            
        # Realizar la predicción
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.success("¡Resultado: **Bestseller**! Este producto reúne las condiciones de mercado idóneas para ser un éxito en ventas.")
        else:
            st.info("Resultado: **Producto Regular**. El perfil nutricional o comercial analizado proyecta un nivel de ventas estándar.")

except Exception as e:
    st.error(f"Error al inicializar la aplicación o cargar los modelos: {e}")
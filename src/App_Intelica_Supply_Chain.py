import streamlit as st
from Intelica_Backend_0 import *



# === Streamlit App ===
def main():
    data = Data()
    st.title("🔍 Análisis de precios con TDA + Markov + ML")

    # Sidebar de selección
    product_selection = st.sidebar.selectbox(
    "Selecciona un producto:",
    data.get_products()
)
    freq_selection = st.sidebar.selectbox(
    "Frecuencia de remuestreo:",
    ["W", "D", "M"], # Semanal, Diario, Mensual
    index=0 # Por defecto Semanal
)
    steps_selction = st.slider("Seleccione el número de pasos", 1, 10)

    if st.sidebar.button("Ejecutar Pronóstico"):
        st.header(f"Pronóstico para {product_selection.capitalize()}")

        arima_results, tda_results = trial(product = product_selection, freq= freq_selection, steps=steps_selction)

        st.subheader("Resultados Modelo ARIMA")
        if arima_results:
            st.write("Métricas de rendimiento:")
            st.json(arima_results['Metrics'])
            st.write(f"**Ganancia estimada sobre el pronóstico:** {arima_results['Expected_Profit_Pct']:.2f}%")
            
            # --- Mostrar Probabilidad de Markov para ARIMA ---
            if arima_results["Markov_Next_State_Proba"]:
                ARIMA_prob, ARIMA_state = arima_results['Markov_Next_State_Proba']
                st.markdown(f"**Pronóstico de Markov (ARIMA):** Lo más probable es que el precio **'{ARIMA_state}'** con una probabilidad de **{ARIMA_prob:.2%}**.")
            else:
                st.info("Información de Markov para ARIMA no disponible o con error.")
            # --- Fin Probabilidad de Markov para ARIMA ---
            
            st.write("Gráfica del pronóstico:")
            st.pyplot(arima_results['Forecast_Plot'])

        st.subheader("Resultados Modelo TDA")
        if tda_results:
            st.write("Métricas de rendimiento:")
            st.json(tda_results['Metrics'])
            st.write(f"**Ganancia estimada sobre las predicciones:** {tda_results['Expected_Profit_Pct']:.2f}%")
            
            # --- Mostrar Probabilidad de Markov para TDA ---
            if tda_results["Markov_Next_State_Proba"]:
                TDA_prob, TDA_state = tda_results['Markov_Next_State_Proba']
                st.markdown(f"**Pronóstico de Markov (TDA):** Lo más probable es que el precio **'{TDA_state}'** con una probabilidad de **{TDA_prob:.2%}**.")
            else:
                st.info("Información de Markov para TDA no disponible o con error.")
            # --- Fin Probabilidad de Markov para TDA ---
            
            st.write("Gráfica de predicciones (test set):")
            st.pyplot(tda_results['Forecast_Plot'])
        else:
            st.warning("No se pudo ejecutar el modelo TDA. Revisa los parámetros o los datos.")
main()


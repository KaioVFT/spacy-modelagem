import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

from preprocess import processar_texto, NLP

# --- CONFIGURAÇÕES DE CAMINHO ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"

@st.cache_resource
def carregar_classificador() -> tuple:
    """Carrega o modelo de classificação de notícias e o vetorizador."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    modelo = joblib.load(MODEL_PATH)
    vetorizador = joblib.load(VECTORIZER_PATH)
    return modelo, vetorizador

@st.cache_resource
def carregar_pipeline_sentimento():
    """Carrega o pipeline de análise de sentimentos da Hugging Face."""
    from transformers import pipeline
    # Usando modelo público e estável
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

modelo_classificador, vetorizador = carregar_classificador()
pipeline_sentimento = carregar_pipeline_sentimento()

st.set_page_config(page_title="Análise de Notícias", page_icon="🔬", layout="centered")
st.title("🔬 Ferramenta de Análise de Notícias")
st.markdown("Navegue pelas abas abaixo para usar as diferentes funcionalidades de Processamento de Linguagem Natural.")

tab1, tab2 = st.tabs(["Detector de Fake News", "Análises Avançadas (POS e Sentimento)"])

# --- ABA 1: DETECTOR DE FAKE NEWS ---
with tab1:
    st.header("Classificação de Notícias (Verdadeira ou Falsa)")
    st.markdown("Cole o título de uma notícia para descobrir se ela é provavelmente verdadeira ou falsa.")

    if modelo_classificador is None or vetorizador is None:
        st.error("⚠️ Modelo de classificação não encontrado! Treine o modelo primeiro (`python src/train.py`).")
    else:
        texto_noticia_classificar = st.text_area("Texto da notícia para classificar", height=150, key="text_classifier")
        if st.button("Classificar", type="primary", use_container_width=True, key="btn_classifier"):
            if texto_noticia_classificar.strip():
                texto_processado = processar_texto(texto_noticia_classificar)
                vetor_texto = vetorizador.transform([texto_processado])
                predicao = modelo_classificador.predict(vetor_texto)
                probabilidade = modelo_classificador.predict_proba(vetor_texto)
                conf_falsa, conf_verdadeira = probabilidade[0]

                st.markdown("---")
                st.subheader("Resultado da Classificação")
                if predicao[0] == 1:
                    st.success("✅ Resultado: VERDADEIRA")
                    st.progress(conf_verdadeira, text=f"Confiança: {conf_verdadeira:.1%}")
                else:
                    st.error("❌ Resultado: FALSA")
                    st.progress(conf_falsa, text=f"Confiança: {conf_falsa:.1%}")
            else:
                st.warning("Por favor, insira um texto para classificar.")

# --- ABA 2: ANÁLISES AVANÇADAS ---
with tab2:
    st.header("Análise de Texto (POS Tagging e Sentimento)")
    st.markdown("Cole um texto qualquer para extrair suas classes gramaticais (POS Tagging) e analisar seu sentimento.")

    texto_analise_avancada = st.text_area("Texto para análise avançada", height=150, key="text_advanced")
    if st.button("Analisar Texto", type="primary", use_container_width=True, key="btn_advanced"):
        if texto_analise_avancada.strip():
            doc = NLP(texto_analise_avancada)
            st.markdown("---")
            st.subheader("📊 Análise de Classes Gramaticais (Part-of-Speech Tagging)")

            pos_data = [(token.text, token.lemma_, token.pos_, spacy.explain(token.pos_)) for token in doc if not token.is_punct]
            df_pos = pd.DataFrame(pos_data, columns=["Token", "Lema", "POS Tag", "Descrição"])
            st.dataframe(df_pos)

            st.markdown("#### Frequência das Classes Gramaticais")
            fig, ax = plt.subplots(figsize=(8, 4))  # Tamanho reduzido
            pos_counts = df_pos['POS Tag'].value_counts()
            sns.barplot(x=pos_counts.index, y=pos_counts.values, ax=ax, palette="viridis")
            ax.set_title("Contagem de Classes Gramaticais (POS)")
            ax.set_ylabel("Frequência")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("😊 Análise de Sentimento")
            resultado_sentimento = pipeline_sentimento(texto_analise_avancada)
            # O modelo retorna labels como '1 star', '2 stars', ..., '5 stars'
            label_map = {
                '1 star': 'Muito Negativo',
                '2 stars': 'Negativo',
                '3 stars': 'Neutro',
                '4 stars': 'Positivo',
                '5 stars': 'Muito Positivo'
            }
            sentimento = label_map.get(resultado_sentimento[0]['label'], resultado_sentimento[0]['label'])
            score = resultado_sentimento[0]['score']

            if 'Positivo' in sentimento:
                st.success(f"Sentimento: {sentimento} (Confiança: {score:.1%})")
            elif 'Negativo' in sentimento:
                st.error(f"Sentimento: {sentimento} (Confiança: {score:.1%})")
            else:
                st.info(f"Sentimento: {sentimento} (Confiança: {score:.1%})")

            fig_sent, ax_sent = plt.subplots(figsize=(8, 4))  # Tamanho reduzido
            sns.barplot(x=[sentimento], y=[score], ax=ax_sent, palette="rocket")
            ax_sent.set_title("Resultado da Análise de Sentimento")
            ax_sent.set_ylabel("Confiança")
            ax_sent.set_ylim(0, 1)
            st.pyplot(fig_sent)
        else:
            st.warning("Por favor, insira um texto para analisar.")
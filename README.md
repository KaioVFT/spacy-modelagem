# Classificador de Notícias Falsas com spaCy

Este projeto implementa um modelo de Machine Learning para classificar notícias como verdadeiras ou falsas, utilizando `spaCy` para pré-processamento de texto e `Scikit-learn` para o modelo. Inclui uma interface web interativa com Streamlit e funcionalidades extras de análise gramatical (POS Tagging) e análise de sentimento usando Hugging Face Transformers.

## Tecnologias Utilizadas
- Python 3.10+
- spaCy
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Transformers (Hugging Face)
- Matplotlib & Seaborn

## Estrutura do Projeto

```
spacy-modelagem/
├── data/
│   └── noticias.csv                       # Dataset de treinamento
├── models/
│   ├── logistic_regression_model.joblib   # Modelo treinado
│   └── tfidf_vectorizer.joblib            # Vetorizador treinado
├── src/
│   ├── preprocess.py                      # Pré-processamento de texto com spaCy
│   ├── train.py                           # Treinamento do modelo
│   ├── predict.py                         # Previsão via terminal
│   └── app.py                             # Interface web com Streamlit
├── .gitignore
├── README.md
└── requirements.txt
```

## Funcionalidades

- **Treinamento de modelo de classificação de notícias (falsas/verdadeiras)**
- **Interface web interativa com Streamlit**
  - Detector de Fake News
  - Análise gramatical (POS Tagging)
  - Análise de sentimento (usando modelo público do Hugging Face)
- **Previsão via terminal**
- **Visualização de frequência de classes gramaticais**
- **Barra de confiança para classificação e sentimento**

## Como usar

### 1. Instale as dependências

```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
```

### 2. Treine o modelo

```bash
python src/train.py
```

Os arquivos do modelo serão salvos na pasta `models/`.

### 3. Execute a interface web

```bash
streamlit run src/app.py
```

### 4. Faça previsões via terminal (opcional)

```bash
python src/predict.py
```

## Observações

- Certifique-se de que o arquivo `noticias.csv` está presente na pasta `data/`.
- Os arquivos `.joblib` do modelo e do vetorizador serão gerados automaticamente após o treinamento.
- O modelo de sentimento utilizado é público: `nlptown/bert-base-multilingual-uncased-sentiment`.
- O projeto está organizado para facilitar manutenção e expansão.

---

Projeto `spacy-modelagem`
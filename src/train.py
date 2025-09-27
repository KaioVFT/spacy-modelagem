import pandas as pd
import joblib
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from preprocess import processar_texto

# --- CONFIGURAÇÕES ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "noticias.csv"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"

MODEL_DIR.mkdir(exist_ok=True)

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Parâmetros do vetorizador e do modelo
TFIDF_PARAMS = dict(max_features=5000, ngram_range=(1, 2))
LOGREG_PARAMS = dict(solver='liblinear', random_state=42, class_weight='balanced')

def carregar_dados() -> pd.DataFrame | None:
    """Carrega e pré-processa o conjunto de dados."""
    try:
        df = pd.read_csv(DATASET_PATH, delimiter=';', engine='python')
    except FileNotFoundError:
        logging.error(f"Arquivo '{DATASET_PATH.name}' não encontrado na pasta 'data'.")
        return None

    df = df[['title', 'real']].copy()
    df.rename(columns={'title': 'text', 'real': 'label'}, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    df['label'] = df['label'].astype(int)

    logging.info(f"{len(df)} notícias carregadas.")
    logging.info(f"Distribuição das classes: {df['label'].value_counts(normalize=True).to_dict()}")
    return df

def treinar_modelo() -> None:
    """Treina e avalia o modelo de regressão logística."""
    df = carregar_dados()
    if df is None:
        return

    # Pré-processamento dos textos
    df['processed_text'] = df['text'].apply(processar_texto)
    X = df['processed_text'].fillna('')
    y = df['label']

    # Separação em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vetorização dos textos
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Treinamento do modelo
    model = LogisticRegression(**LOGREG_PARAMS)
    model.fit(X_train_vec, y_train)

    # Avaliação do modelo
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Acurácia: {acc:.4f}")
    logging.info("Relatório de classificação:\n" + classification_report(y_test, y_pred, target_names=['Falsa', 'Verdadeira']))

    # Salvando o modelo e o vetorizador
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Modelo salvo em: {MODEL_PATH}")
    logging.info(f"Vetorizador salvo em: {VECTORIZER_PATH}")

if __name__ == '__main__':
    treinar_modelo()
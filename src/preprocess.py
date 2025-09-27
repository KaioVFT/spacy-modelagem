import spacy

def carregar_modelo():
    """Carrega o modelo spaCy para português."""
    try:
        return spacy.load("pt_core_news_lg")
    except OSError as e:
        raise RuntimeError("Modelo 'pt_core_news_lg' não encontrado. Execute: python -m spacy download pt_core_news_lg") from e

NLP = carregar_modelo()

def processar_texto(texto: str) -> str:
    """
    Pré-processa o texto: minúsculas, lematização, remoção de stopwords, pontuação e tokens não alfabéticos.
    """
    if not isinstance(texto, str):
        return ""
    doc = NLP(texto.lower())
    lemmas = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(lemmas)
import spacy

def carregar_modelo():
    """Carrega o modelo spaCy para inglês."""
    try:
        return spacy.load("en_core_web_lg")
    except OSError as e:
        raise RuntimeError("Modelo 'en_core_web_lg' não encontrado. Execute: python -m spacy download en_core_web_lg") from e

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
import joblib
import logging
from pathlib import Path

from preprocess import processar_texto

# --- CONFIGURAÇÕES DE CAMINHO ---
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ClassificadorDeNoticias:
    """Classe para carregar e usar o modelo de classificação de notícias."""
    def __init__(self, model_path: Path, vectorizer_path: Path):
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            logging.info("Classificador pronto para uso.")
        except FileNotFoundError:
            logging.error("Arquivos de modelo ou vetorizador não encontrados. Execute 'python src/train.py' primeiro.")
            raise

    def classificar(self, texto: str) -> str:
        """
        Classifica um texto como 'Verdadeira' ou 'Falsa'.
        Retorna o resultado e a confiança.
        """
        texto_processado = processar_texto(texto)
        vetor_texto = self.vectorizer.transform([texto_processado])
        predicao = self.model.predict(vetor_texto)
        probabilidade = self.model.predict_proba(vetor_texto)
        confianca = probabilidade[0][predicao[0]]
        resultado = "Verdadeira" if predicao[0] == 1 else "Falsa"
        return f"Resultado: {resultado} (Confiança: {confianca:.2%})"

if __name__ == '__main__':
    classificador = ClassificadorDeNoticias(MODEL_PATH, VECTORIZER_PATH)
    print("\n--- Detector de Notícias Falsas Interativo ---")
    print("Digite o título de uma notícia para analisar.")
    print("Digite 'sair' ou 'exit' para fechar o programa.")

    while True:
        noticia_usuario = input("\n➡️ Digite a notícia: ")
        if noticia_usuario.lower() in ['sair', 'exit']:
            print("Encerrando o programa. Até mais!")
            break
        if not noticia_usuario.strip():
            print("Por favor, digite algum texto.")
            continue
        resultado_classificacao = classificador.classificar(noticia_usuario)
        print(f"📰 Análise: {resultado_classificacao}")
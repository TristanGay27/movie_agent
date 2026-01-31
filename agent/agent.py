import os
import pickle
import pandas as pd

import torch
from langchain.agents import create_agent
from langchain.messages import SystemMessage,HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from flask import Flask, jsonify, request, send_file

from langchain.chat_models import init_chat_model

from classifier import Classif

app = Flask(__name__)

YOUR_KEY = "to_replace"

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = Classif(input_dim=3000, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("genre_model.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def load_faiss_index(csv_path="movies_metadata.csv"):
    """
    Charge movies_metadata.csv (Kaggle) et crée un index FAISS.
    """

    # Lecture du CSV (low_memory=False évite des bugs de type)
    df = pd.read_csv(csv_path, low_memory=False)

    # On garde uniquement les lignes exploitables
    df = df[["title", "overview"]].dropna()
    df = df[df["overview"].str.len() > 20]
    #df = df.sample(1000, random_state=42) #pour le début

    docs = [
        Document(
            page_content=row["overview"],
            metadata={"title": row["title"]}
        )
        for _, row in df.iterrows()
    ]

    embeddings = OpenAIEmbeddings()  # utilise OPENAI_API_KEY
    index = FAISS.from_documents(docs, embeddings)

    return index

def new_agent():
    os.environ["OPENAI_API_KEY"] = YOUR_KEY
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    llm = init_chat_model(
        model="upstage/solar-pro-3:free",
        model_provider="openai",
        temperature=0.7,
        max_tokens=2048,
    )

    faiss_index = load_faiss_index()

    @tool
    def movie_search(query: str, k: int = 5) -> str:
        """Recherche des films similaires dans la base de données.
        
        Args:
            query: Le film ou thème à rechercher
            k: Nombre de résultats (défaut: 5)
        """
        print("Utilisation de movie_search")
        results = faiss_index.similarity_search(query, k=k)
        formatted = "\n".join(
                f"- {r.metadata.get('title', 'Titre inconnu')}: {r.page_content[:200]}..."
                for r in results)        
        return f"Résultats pour '{query}':\n{formatted}"

    @tool
    def classify_genre(plot: str) -> str:
        """Classifie le genre d'un film à partir de son résumé (plot).
        
        Args:
            plot: Le résumé du film à classifier
        """
        print("Utilisation de classify_genre")

        # TF-IDF vectorisation
        X_plot = vectorizer.transform([plot]).toarray()
        X_tensor = torch.tensor(X_plot, dtype=torch.float32).to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(outputs, dim=1).item()
        
        genre = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = probabilities[0][predicted_idx].item()
    

        return f"Genre prédit: **{genre}** (confiance: {confidence:.1%})"

    tools = [movie_search, classify_genre]

    agent = create_agent(
        tools=tools,
        model=llm,
        system_prompt=SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": """
                            Tu es un expert en cinéma.

                            Tu dois
                            1. Identifier le film décrit par l'utilisateur.
                            2. Classe le genre uniquement selon classify genre, en précisant la confiance
                            3. Proposer 5 films similaires en utilisant uniquement les résultats de movie_search

                            Important :
                                - Fais obligatoirement entre 1 et 2 appels par outils
                                - La base de données de films est en ANGLAIS.
                                - Si la question de l’utilisateur est en français, tu DOIS d’abord traduire le synopsis en anglais
                                avant toute recherche ou appel d’outil.
                                - Les réponses finales doivent rester en français.
                            
                            Réponds de manière structurée, claire et pas trop longue.
                            """,
                }])
    )

    return agent


agent = new_agent()
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    response = agent.invoke({
        "messages": [HumanMessage(content=text)]
    })

    return jsonify({
        "response": response["messages"][-1].content
    })

# ----------------- RUN -----------------
if __name__ == "__main__":
    app.run(port=5000, debug=False, host="0.0.0.0")
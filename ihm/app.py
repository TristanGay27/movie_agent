import streamlit as st
import requests

API_URL = "http://agent:5000/predict"


st.set_page_config(page_title="🎬 Agent IA - Reco Films", layout="centered")

st.title("🎬 Agent IA de recommandation de films")
st.write("Décris un film ou donne un synopsis, l'agent te répond.")

# Zone de texte utilisateur
user_input = st.text_area(
    "Synopsis / question",
    placeholder="Un film où on entre dans les rêves pour manipuler la réalité...",
    height=150
)

# Bouton
if st.button("Analyser"):
    if not user_input.strip():
        st.warning("Merci d'entrer une description.")
    else:
        with st.spinner("Analyse..."):
            r = requests.post(API_URL, json={"text": user_input})
            if r.status_code == 200:
                st.subheader("Réponse de l'agent")
                st.write(r.json()["response"])
            else:
                st.error("Erreur API")

    # if not user_input.strip():
    #     st.warning("Merci d'entrer une description.")
    # else:
    #     with st.spinner("L'agent réfléchit..."):
    #         response = agent.invoke({"messages": [HumanMessage(user_input)]})

    #     st.subheader("🤖 Réponse de l'agent")
    #     st.write(response["messages"][-1].content)
    #     st.write(response)


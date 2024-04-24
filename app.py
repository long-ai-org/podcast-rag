import os

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.zilliz import Zilliz
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from streamlit_chat import message

template = """
        Beantworten Sie die Frage des Nutzers am Ende des Textes anhand der folgenden Informationen. 
        Bei den Informationen handelt es sich um transkribierte Ausschnitte aus verschiedenen 
        Podcast-Episoden über das Arbeiten in China. Die Fragen der Nutzer sollen nun nur auf der 
        Grundlage der Antworten der Podcast-Gäste aus den Podcasts beantwortet werden. 
        Wenn in den Informationen kein relevantes Wissen zur Beantwortung der Frage vorhanden ist,
        sagen Sie bitte "Ich habe kein Podcast-Wissen zu dieser Frage". Andernfalls sollte die 
        Ausgabe in Stichpunkten als Aufzählungspunkte erfolgen (jeder Aufzählungspunkt in einer Zeile) und der 
        Name des Podcast-Gastes und der Name des Podcasts sollten in Klammern nach jedem Punkt im 
        Format "(PODCAST NAME - PODCAST GAST)" stehen. Darüber hinaus sollte kein Ausgabetext angezeigt werden.
        Wenn der Nutzer in Englisch seine Frage stellt, soll die Antwort auch ins Englische übersetzt werden.

{context}

Frage: {question}

Hilfreiche Antwort:
"""
custom_rag_prompt = PromptTemplate.from_template(template)


load_dotenv()

embeddings = AzureOpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDINGS_MODEL"))
llm = AzureChatOpenAI(
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT_NAME"),
    temperature=0,
    streaming=True,
    max_tokens=4096,
)
connection_args = {
    "uri": os.environ.get("ZILLIZ_CLOUD_URI"),
    "token": os.environ.get("ZILLIZ_CLOUD_API_KEY"),
}

COLLECTION_NAME = "podcast_fragen_antworten_chunks"
vectorstore = Zilliz(
    embedding_function=embeddings,
    connection_args=connection_args,
    collection_name=COLLECTION_NAME,
)
retriever = vectorstore.as_retriever()


def main():
    st.set_page_config(page_title="The China PodcastBot", layout="wide")
    st.markdown(
        """
        The China PodcastBot is an AI that allows access to knowledge from various podcasts about living and working in China.
        Ask a China-related business question, and the AI will display all the answers from podcast guests on various podcasts.
    """
    )

    if "response" not in st.session_state:
        st.session_state["responses"] = ["How can I assist you today?"]

    if "requests" not in st.session_state:
        st.session_state["requests"] = []

    response_container = st.container()
    text_container = st.container()

    with text_container:
        query = st.chat_input("Please enter a question.", key="input")
        if query:
            with st.spinner("typing..."):
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | custom_rag_prompt
                    | llm
                )

            response = rag_chain.invoke(query).content

            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state["responses"]:

            for i in range(len(st.session_state["responses"])):
                message(st.session_state["responses"][i], key=str(i))
                if i < len(st.session_state["requests"]):
                    message(
                        st.session_state["requests"][i],
                        is_user=True,
                        key=str(i) + "_user",
                    )


if __name__ == "__main__":
    main()

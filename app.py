import os

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.zilliz import Zilliz
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

template = """
        Beantworten Sie die Frage des Nutzers am Ende des Textes anhand der folgenden Informationen.
        Bei den Informationen handelt es sich um transkribierte Ausschnitte aus verschiedenen
        Podcast-Episoden über das Arbeiten in China. Die Fragen der Nutzer sollen nun nur auf der
        Grundlage der Antworten der Podcast-Gäste aus den Podcasts beantwortet werden. 
        Wenn in den Informationen kein relevantes Wissen zur Beantwortung der Frage vorhanden ist,
        sagen Sie bitte "Ich habe kein Podcast-Wissen zu dieser Frage" if question was asked in german 
        and "I have no podcast knowledge on this question" if question asked in English. Andernfalls sollte die
        Ausgabe in Stichpunkten als Aufzählungspunkte erfolgen (jeder Aufzählungspunkt in einer Zeile)
        und der Name des Podcast-Gastes und der Name des Podcasts sollten in Klammern nach jedem
        Punkt im Format "(PODCAST NAME - PODCAST GAST)" stehen. Am Anfang der Antwort soll hierbei
        stehen "Ich habe folgenden Informationen gefunden:" Darüber hinaus sollte kein Ausgabetext angezeigt werden.
        Wenn der Nutzer in Englisch seine Frage stellt, soll auch die Antwort in Englische zurückgegeben werden, 
        auch die Zitate aus dem Podcast müssen dann ins Englische übersetzt werden.

        Before answering the question perform the following steps:
        1. Determine which language the user's question is in.
        2. Answer the user's question using the language of the user's question.

        {context}

        Frage: {question}

        Useful Answer:
"""
custom_rag_prompt = PromptTemplate.from_template(template)

llm = AzureChatOpenAI(
    deployment_name=os.environ.get("AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT_NAME"),
    temperature=0,
    streaming=True,
    max_tokens=4096,
)


embeddings = AzureOpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDINGS_MODEL"))
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
    st.set_page_config(page_title="The China PodcastBot", layout="centered")
    st.title("The China PodcastBot: Ask questions about working in China")
    st.markdown(
        """
        The China PodcastBot is an AI that allows access to knowledge from various podcasts about living and working in China.
        Ask a China-related business question, and the AI will display all the answers from podcast guests on various podcasts.
    """
    )

    if "message" not in st.session_state:
        st.session_state.message = []

    for message in st.session_state.message:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Please enter a question."):
        st.session_state.message.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
            )

            response = st.write_stream(rag_chain.stream(prompt))
        st.session_state.message.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

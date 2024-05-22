import os

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.zilliz import Zilliz
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

template = """
        Answer the user's question at the end of the text using the following information.
        The information are transcribed excerpts from various podcast episodes about working in China.
        The user's questions will now be answered based only on the podcast guests' answers from the podcasts.
        If there is no relevant knowledge in the information to answer the question, please say 
        “I have no podcast knowledge on this question”. Otherwise, the output should be in bullet
        points (each bullet point on one line) and the name of the podcast guest and the name of the 
        podcast episode should be in brackets after each bullet point in the format “(PODCAST EPISODE - PODCAST GUEST)”.
        The answer should start with “I found the following information:” 
        No further output text should be displayed.

        {context}

        Question: {question}

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

COLLECTION_NAME = "podcast_questions_answers_chunks"
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
        The China PodcastBot is an AI chatbot that allows access to knowledge from various podcasts about living and working in China.
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

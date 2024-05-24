import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following information:

{context}

---

Answer the question based on the above context: {question}

---

Do not hallucinate.
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.", nargs='?', default='')
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Initialize conversation history
    conversation_history = []

    # Start the conversation with a greeting
    print("Hi! How can I help you?")
    conversation_history.append("Assistant: Hi! How can I help you with you?")

    while True:
        # Prompt the user for the query
        query_text = input("User: ")

        if query_text.lower() in ["quit", "exit", "bye"]:
            print("Assistant: Goodbye!")
            break

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = Ollama(model="llama3")
        response_text = model.invoke(prompt)
        print(response_text)

        # Add the current query and response to the conversation history
        conversation_history.append(f"User: {query_text}")
        conversation_history.append(f"Assistant: {response_text}")
    
    return "\n".join(conversation_history)


if __name__ == "__main__":
    main()

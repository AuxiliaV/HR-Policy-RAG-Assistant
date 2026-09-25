import os
from dotenv import load_dotenv
openi_api_key = os.getenv("OPENAI_API_KEY")

# Load the PDF

from langchain_community.document_loaders import PyPDFLoader

work_dir = os.getcwd()
work_dir = work_dir.replace("\\","/")

# path = work_dir+"/data/acme_leave_policy.pdf"

pdf_path = [ work_dir+"/data/acme_expense_policy.pdf",
            work_dir+"/data/acme_leave_policy.pdf",
            work_dir+"/data/acme_remote_work_policy.pdf"]


pages = []

for path in pdf_path:
    for page in PyPDFLoader(path).lazy_load():
        page.metadata["source_name"] = os.path.basename(path).replace(".pdf","").replace("_"," ").title()
        page.metadata["page_num"] = page.metadata["page"]+1

        page.page_content = "\n".join(line for line in page.page_content.splitlines()
                                  if "Internal use only" not in line and not line.strip().startswith("Page ")) 
        # Data cleaning: removing the empty space and footer content

        pages.append(page)


### Prepare Chunks ###

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = splitter.split_documents(pages)  # metadata is copied to every chunk

# print(len(chunks), "chunks")
# print(chunks[17].metadata["source_name"], "| page", chunks[17].metadata["page_num"])
# print(chunks[17].page_content[:200])


### Embedding and Storing in Vector DB ###

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
# print("Index ready")


### Function for Citation ###

def label(doc):
    # The citation label, e.g. 'Acme Leave Policy, p.2'
    return f"{doc.metadata['source_name']}, p.{doc.metadata['page_num']}"


### Langchain Setup ###

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR policy assistant. Answer using ONLY the sources below.\n\n"
     "Rules:\n"
     "- After every statement, add the source label in square brackets exactly as written, "
     "for example [Acme Leave Policy, p.3].\n"
     "- Mention any conditions or exceptions.\n"
     "- If the sources do not contain the answer, reply exactly: I couldn't find this in the provided documents.\n"
     "- Keep the answer short.\n\n"
     "Sources:\n{context}"),

    ("human", "{question}"),
])

### Function for Question and Answer ###

def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{label(d)}]\n{d.page_content}" for d in docs)
    messages = prompt.invoke({"context": context, "question": question})
    answer = llm.invoke(messages).content

    return answer

chat_history = []  # list of (question, answer) pairs

def ask_with_memory(question):
    search_question = question
    if chat_history:
        history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:])
        rewrite_request = (
            "Rewrite the latest question as a standalone question that makes sense without the chat history. "
            "Do NOT answer it.\n\n"
            f"Chat history:\n{history_text}\n\nLatest question: {question}\n\nStandalone question:"
        )
        search_question = llm.invoke(rewrite_request).content
        print("(searching for:", search_question, ")")
    answer = ask(search_question)
    chat_history.append((question, answer))
    return answer

#print("\nPlease Type 'quit' or 'exit' while leaving the chat\n")

def main(Question):
    #while True:

    queries = Question

        #question = input("\nEnter your question: ")
    #if question.lower() in ("quit","exit"):
        #break
    #else:
    ans = ask_with_memory(queries)
    print(ans)

    return ans
        
#q = "paid sick leave for new joinee"
#a = main(q)
#print


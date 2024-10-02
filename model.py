import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Set page configuration
st.set_page_config(page_title="FDA Audit Assistant", page_icon="🏥", layout="wide")

# Load the FAISS index
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vector_store = load_vector_store()

# Initialize the Ollama LLM
llm = Ollama(model="llama2")

# Set up the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Define prompt templates
question_generation_template = """You are an AI assistant specialized in FDA regulations for the pharmaceutical industry. Your task is to create a questionnaire for an FDA Audit based on the regulations in the provided context. Generate 10 questions relevant to the user's specific use case.

After answering these questions, the FDA may give one of three findings: "No findings", "Warning", or "483 issued".

Use the following context to formulate your questions:

{context}

Remember to tailor the questions to the user's specific use case."""

conversation_template = """You are an AI assistant specialized in FDA regulations for the pharmaceutical industry. Your tasks include:

1. Analyzing responses to identify compliance gaps.
2. Suggesting mitigation strategies to address identified gaps.
3. Providing information on FDA regulations and best practices.

Use the following context to inform your responses:

{context}

Answer the following user query directly don't greet the user.: {question}

Remember to tailor your responses to the user's specific queries and use cases."""

# Create chat prompt templates
question_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(question_generation_template),
    HumanMessagePromptTemplate.from_template("Use case for FDA Audit: {question}\n\nPlease generate 10 relevant questions for this FDA Audit use case.")
])

conversation_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(conversation_template),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Create QA chains
question_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": question_prompt}
)

conversation_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": conversation_prompt}
)

st.title("🏥 FDA Audit Assistant")

st.markdown("""
This assistant helps you prepare for FDA audits in the pharmaceutical industry. 
It generates relevant questions based on your use case and provides guidance on compliance issues.
""")

st.sidebar.header("Audit Use Case")
use_case = st.sidebar.text_area("Enter your specific use case for the FDA Audit:", height=100)

if st.sidebar.button("Generate Questionnaire"):
    if use_case:
        with st.spinner("Generating questionnaire..."):
            questionnaire = question_chain.run(use_case)
        st.session_state['questionnaire'] = questionnaire
        st.session_state['conversation_started'] = True
        # Reset chat history for new use case
        st.session_state['chat_history'] = []
    else:
        st.sidebar.error("Please enter a use case.")

if 'questionnaire' in st.session_state:
    st.header("Generated Questionnaire")
    st.write(st.session_state['questionnaire'])

if 'conversation_started' in st.session_state and st.session_state['conversation_started']:
    st.header("Compliance Conversation")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display chat history
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question or provide information about your compliance:")
    
    if user_input:
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            with st.chat_message("assistant"):
                st.markdown("Goodbye! Feel free to start a new session anytime.")
            # Reset session state
            del st.session_state['conversation_started']
            del st.session_state['questionnaire']
            del st.session_state['chat_history']
        else:
            # Add user message to chat history
            st.session_state['chat_history'].append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = conversation_chain.run(user_input)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("To finish the conversation, type 'exit', 'quit', or 'bye'.")
import os
import re
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import Document

# Set page configuration
st.set_page_config(page_title="FDA Audit Assistant", page_icon="🏥", layout="wide")

# List of all PDF files with their part numbers in the name
pdf_files_with_parts = {
    'data/11.pdf': ['11'],
    'data/200.pdf': ['200'],
    'data/201.pdf': ['201'],
    'data/202.pdf': ['202'],
    'data/203.pdf': ['203'],
    'data/205.pdf': ['205'],
    'data/206.pdf': ['206'],
    'data/207.pdf': ['207'],
    'data/208.pdf': ['208'],
    'data/209.pdf': ['209'],
    'data/210.pdf': ['210'],
    'data/211.pdf': ['211'],
    'data/212.pdf': ['212'],
    'data/216.pdf': ['216'],
    'data/225.pdf': ['225'],
    'data/226.pdf': ['226'],
    'data/250.pdf': ['250'],
    'data/251.pdf': ['251'],
    'data/290.pdf': ['290'],
    'data/299.pdf': ['299'],
    'data/312.pdf': ['312'],
    'data/314.pdf': ['314'],
    'data/600.pdf': ['600'],
    'data/601.pdf': ['601'],
    'data/606.pdf': ['606'],
    'data/607.pdf': ['607'],
    'data/610.pdf': ['610'],
    'data/630.pdf': ['630'],
    'data/640.pdf': ['640'],
    'data/660.pdf': ['660'],
    'data/680.pdf': ['680'],
    'data/820.pdf': ['820']
}

# Regex patterns to detect part titles, subparts, and sections
part_pattern = re.compile(r"PART\s+(\d+)\s*—\s*(.*)", re.IGNORECASE)  # Detects "PART 11—ELECTRONIC RECORDS"
subpart_pattern = re.compile(r"Subpart\s+([A-Z])\s*—\s*(.*)", re.IGNORECASE)  # Detects "Subpart A—General Provisions"
section_pattern = re.compile(r"^\s*§\s*\d+\.\d+", re.IGNORECASE | re.MULTILINE)  # Detects "§ 11.1", "§ 11.10" at line starts

# Function to extract the correct part and its title
def extract_correct_part_and_title(text, part_number):
    """
    Extracts the specified part (e.g., Part 11) and its title from the full PDF text.
    
    Args:
        text: The raw text extracted from the PDF.
        part_number: The part number to look for (e.g., '11', '312').
    
    Returns:
        A tuple with the part title and the text of the specified part. If part does not exist, returns (None, None).
    """
    part_text = []
    part_title = None
    lines = text.split("\n")
    in_part = False

    for line in lines:
        # Detect the start of the part and its title (e.g., "PART 11—ELECTRONIC RECORDS")
        match = part_pattern.match(line.strip())
        if match:
            current_part = match.group(1)
            current_title = match.group(2)
            if current_part == part_number:
                part_title = current_title.strip()  # Store the part title
                in_part = True  # We are in the correct part
            else:
                if in_part:
                    break  # We reached a new part, stop collecting
        
        # If we are in the correct part, collect the lines
        if in_part:
            part_text.append(line)
    
    # Check if we found the part, otherwise return None
    if part_title and part_text:
        return part_title, "\n".join(part_text)
    else:
        return None, None  # Part does not exist

# Function to extract subparts and sections from the document text
def extract_subparts_sections_with_titles(part_text):
    """
    Extracts subparts, subpart titles, and sections from the text of a part.
    
    Args:
        part_text: The text of the part.
    
    Returns:
        A dictionary with subpart names, subpart titles, and sections within each subpart.
    """
    subparts = {}
    current_subpart = None
    current_subpart_title = None
    current_sections = []
    
    lines = part_text.split("\n")
    
    for line in lines:
        # Detect subparts (e.g., "Subpart A—General Provisions")
        subpart_match = subpart_pattern.match(line.strip())
        if subpart_match:
            # If we're in a new subpart, save the previous one
            if current_subpart and current_sections:
                subparts[current_subpart] = {
                    "subpart_title": current_subpart_title,
                    "sections": current_sections
                }
            # Start a new subpart
            current_subpart = subpart_match.group(1)
            current_subpart_title = subpart_match.group(2).strip()
            current_sections = []
        
        # Detect section headers at the start of a line
        section_match = section_pattern.match(line)
        if section_match:
            current_sections.append(line.strip())  # Collect the section header only
    
    # Save the last subpart
    if current_subpart and current_sections:
        subparts[current_subpart] = {
            "subpart_title": current_subpart_title,
            "sections": current_sections
        }
    
    # If no subpart was detected, treat it as unnamed sections
    if not subparts and current_sections:
        subparts["Unnamed"] = {
            "subpart_title": "Unnamed",
            "sections": current_sections
        }
    
    return subparts

# Function to chunk each section based on the section numbers
def chunk_sections_by_subpart_with_titles(full_text, subparts):
    """
    Chunk the document into sections based on the subparts and sections detected.
    
    Args:
        full_text: The full text of the part.
        subparts: Dictionary of subparts, subpart titles, and their sections from the document.
    
    Returns:
        A dictionary where subparts map to their corresponding chunked sections.
    """
    chunked_subparts = {}
    
    lines = full_text.split("\n")
    current_section = None
    current_chunk = []
    
    # Iterate through all the subparts and sections
    for subpart, subpart_info in subparts.items():
        chunked_subparts[subpart] = {
            "subpart_title": subpart_info["subpart_title"],
            "sections": {}
        }
        
        for section in subpart_info["sections"]:
            section_parts = section.split()
            if len(section_parts) < 2:
                continue  # Skip this section if it's malformed

            section_number = section_parts[1]
            
            # Escape special characters in section_number for regex
            escaped_section_number = re.escape(section_number)

            # Compile the regex to find the section
            section_pattern = re.compile(rf"^\s*§\s*{escaped_section_number}\s+.*", re.IGNORECASE)

            for line in lines:
                if section_pattern.match(line):
                    if current_section and current_chunk:
                        chunked_subparts[subpart]["sections"][current_section] = "\n".join(current_chunk)
                    current_section = section
                    current_chunk = []
                
                if current_section:
                    current_chunk.append(line)
    
        if current_section and current_chunk:
            chunked_subparts[subpart]["sections"][current_section] = "\n".join(current_chunk)
    
    return chunked_subparts


@st.cache_resource
def build_or_load_vector_store():
    """
    Load the FAISS vector store if it exists, otherwise build it from scratch.
    """
    embeddings = HuggingFaceEmbeddings()
    
    if os.path.exists("faiss_index"):
        # If the FAISS index exists, load it from disk
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        # If the FAISS index does not exist, build it from scratch
        all_extracted_parts = {}
        documents = []

        # Load PDFs and extract parts
        for file_path, part_numbers in pdf_files_with_parts.items():
            loader = PyPDFLoader(file_path)
            pdf_documents = loader.load()
            full_text = "\n".join([doc.page_content for doc in pdf_documents])
            
            for part_number in part_numbers:
                # Extract the correct part and its title
                part_title, correct_part_text = extract_correct_part_and_title(full_text, part_number)
                
                # Skip parts that do not exist
                if not part_title or not correct_part_text:
                    continue
                
                # Extract subparts, subpart titles, and sections from the isolated part text
                subparts = extract_subparts_sections_with_titles(correct_part_text)
                
                # Chunk the document into sections based on the subparts
                chunked_subparts = chunk_sections_by_subpart_with_titles(correct_part_text, subparts)
                
                # Add the chunked subparts and part title to the overall extracted parts
                all_extracted_parts[file_path] = {
                    "part_title": part_title,
                    "subparts": chunked_subparts
                }

                # Create LangChain Document objects
                for subpart, subpart_info in chunked_subparts.items():
                    subpart_title = subpart_info["subpart_title"]
                    for section, content in subpart_info["sections"].items():
                        doc = Document(
                            page_content=content,
                            metadata={
                                "file_path": file_path,
                                "part_title": part_title,
                                "subpart": subpart,
                                "subpart_title": subpart_title,
                                "section": section
                            }
                        )
                        documents.append(doc)

        # Create the FAISS index from documents and save it
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store

# Load or build the FAISS vector store
vector_store = build_or_load_vector_store()

# Initialize the Ollama LLM
llm = Ollama(model="llama2", base_url="http://127.0.0.1:11434")

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

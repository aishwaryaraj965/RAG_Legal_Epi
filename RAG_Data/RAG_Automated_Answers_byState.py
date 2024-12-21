#dependencies
#pip install langchain faiss-cpu InstructorEmbedding sentence-transformers pypdf pandas openpyxl

import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

def main():
    # Specify the root folder containing state-specific subfolders
    root_folder = "/Users/aishwaryaraj/Desktop/CDC FELLOWSHIP/Text Analysis Project/RAG DATA"
    
    # Specify the CSV files
    input_csv = "/Users/aishwaryaraj/Desktop/CDC FELLOWSHIP/Text Analysis Project/RAG setup/questions_RAG.csv"
    template_csv = "/Users/aishwaryaraj/Desktop/CDC FELLOWSHIP/Text Analysis Project/RAG setup/Medicolegal-Death-Investigation-(MLDI)-Phase-I-RAG_Answers.csv"  # Update with correct path
    
    # Read questions from the CSV file
    questions_df = pd.read_csv(input_csv)
    question_map = questions_df.set_index("Variable label").T.to_dict()

    #ensure response folder exists
    responses_folder = "/Users/aishwaryaraj/Desktop/CDC FELLOWSHIP/Text Analysis Project/RAG setup/Responses"
    os.makedirs(responses_folder, exist_ok=True)
    
    # Iterate through each state folder
    for state_folder in os.listdir(root_folder):
        state_path = os.path.join(root_folder, state_folder)
        
        # Ensure it's a directory
        if not os.path.isdir(state_path):
            continue
        
        print(f"\nProcessing state: {state_folder}\n")
        
        # Get all PDF files for the state
        pdf_files = [os.path.join(state_path, file) for file in os.listdir(state_path) if file.endswith(".pdf")]

        # Initialize an empty list to hold all documents for the state
        all_documents = []

        # Load and process each PDF
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            all_documents.extend(documents)

        # Split the text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(all_documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create a vector store
        db = FAISS.from_documents(docs, embeddings)

        # Load the local LLM using Ollama
        llm = Ollama(model="mistral")

        # Create the RetrievalQA chain
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

        # Load the answer template for this state
        state_df = pd.read_csv(template_csv)
        
        # Automate asking questions for the current state
        for variable_label, question_data in question_map.items():
            question = question_data["Question text"]
            question_type = question_data["Question type"]
            value_label = question_data["Value label"]

            print(f"Asking: {question} ({variable_label})")
            
            # Define the prompt template
            prompt_template = """
            You are answering questions based on retrieved context from state-specific legal documents.
            - Only use information retrieved from the documents.
            - Cite the sources in Bluebook legal format as follows: "[Document Title, p. XX]."
            - If the context does not provide sufficient information, respond with: "NA"
            - Question: {question}
            """
            
            # Construct the full prompt
            options = f"Options: {', '.join(value_label.split(','))}" if pd.notnull(value_label) else "Free-form response expected."
            question_with_context = f"{prompt_template.format(question=question)} {options}"
            
            # Get the response
            response = qa.run(question_with_context)

            # Find the most relevant document and add citation
            retrieved_docs = db.similarity_search(question, k=1)
            citation = f"[{retrieved_docs[0].metadata.get('source', 'Unknown Title')}, p. {retrieved_docs[0].metadata.get('page', 'Unknown Page')}]" if retrieved_docs else "No citation available"

            # Update the appropriate columns in the state dataframe
            response_col = f"{variable_label}"
            annotation_col = f"{variable_label} annotation"
            pincite_col = f"{variable_label} pincite"

            # Ensure the columns exist in the DataFrame
            if response_col not in state_df.columns:
                state_df[response_col] = ""
            if annotation_col not in state_df.columns:
                state_df[annotation_col] = ""
            if pincite_col not in state_df.columns:
                state_df[pincite_col] = ""

            # Populate the columns
            state_df.loc[0, response_col] = response
            state_df.loc[0, annotation_col] = f"{response} Supporting sources: {citation}"
            state_df.loc[0, pincite_col] = citation
        
        # Save responses for the current state as a separate file
        state_output_path = os.path.join(responses_folder, f"{state_folder}_responses.csv")  # Update with desired folder
        state_df.to_csv(state_output_path, index=False)
        print(f"Responses for {state_folder} saved to {state_output_path}")

if __name__ == "__main__":
    main()

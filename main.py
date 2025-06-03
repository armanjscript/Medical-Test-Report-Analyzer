import streamlit as st
from PIL import Image
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import re
import logging
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract path (adjust for your system)
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
        logger.error("Tesseract path configuration failed")

# State definition for LangGraph
class MedicalAgentState(TypedDict):
    image: Any
    extracted_text: str
    llm_response: str
    error: str

# Initialize Ollama LLM (e.g., Qwen 2.5)
llm = OllamaLLM(model="qwen2.5:latest", num_gpu=1)  # Replace with your preferred medical LLM

# Define the prompt template for medical interpretation
prompt_template = ChatPromptTemplate.from_template(
    """
    You are a medical expert AI. Below is the extracted text from a medical test report. Analyze the text, interpret the results, and provide a clear, concise explanation of the health status for a non-medical user. Highlight any abnormal results, explain their potential implications, and suggest next steps (e.g., consult a doctor). If any information is unclear or incomplete, note it and avoid making assumptions. define the origin of each terms completely without more details.

    Extracted Text:
    {extracted_text}

    Response format:
    **Health Status Summary**:
    [Your summary here]

    **Abnormal Results** (if any):
    [List abnormal results with explanations]

    **Recommendations**:
    [Next steps or advice]
    """
)

# Node 1: Extract text from image using OCR
def extract_text_from_image(state: MedicalAgentState) -> MedicalAgentState:
    try:
        logger.info("Extracting text from image...")
        image = state["image"]
        extracted_text = pytesseract.image_to_string(image, lang="eng")
        # Clean extracted text (remove extra whitespace, newlines)
        extracted_text = re.sub(r'\s+', ' ', extracted_text.strip())
        print(extracted_text)
        if not extracted_text:
            state["error"] = "No text could be extracted from the image."
            logger.warning("No text extracted from image")
        else:
            state["extracted_text"] = extracted_text
            logger.info("Text extracted successfully")
    except Exception as e:
        state["error"] = f"Error during text extraction. Please ensure Tesseract OCR is installed: {str(e)}"
        logger.error(f"Text extraction failed: {str(e)}")
    return state

# Node 2: Process extracted text with LLM
def process_with_llm(state: MedicalAgentState) -> MedicalAgentState:
    if state.get("error"):
        return state
    try:
        logger.info("Processing text with LLM...")
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"extracted_text": state["extracted_text"]})
        print(response)
        state["llm_response"] = response
        logger.info("LLM processing completed")
    except Exception as e:
        state["error"] = f"Error during LLM processing: {str(e)}"
        logger.error(f"LLM processing failed: {str(e)}")
    return state

# Define the LangGraph workflow
def build_graph():
    workflow = StateGraph(MedicalAgentState)
    
    # Add nodes
    workflow.add_node("extract_text", extract_text_from_image)
    workflow.add_node("process_llm", process_with_llm)
    
    # Define edges
    workflow.add_edge("extract_text", "process_llm")
    workflow.add_edge("process_llm", END)
    
    # Set entry point
    workflow.set_entry_point("extract_text")
    
    return workflow.compile()

# Streamlit UI
def main():
    st.title("Medical Test Report Analyzer")
    st.write("Upload a medical test report image to extract and interpret the results.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Test Report", use_container_width=True)
        
        # Initialize state
        state = MedicalAgentState(
            image=image,
            extracted_text="",
            llm_response="",
            error=""
        )
        
        # Run the LangGraph workflow
        try:
            graph = build_graph()
            result = graph.invoke(state)
            
            # Display results
            if result.get("error"):
                st.error(result["error"])
                if "Tesseract OCR" in result["error"]:
                    st.info("Note: You need to install Tesseract OCR on your system. On Linux/Mac: 'brew install tesseract' or 'sudo apt install tesseract-ocr'. On Windows: download from https://github.com/UB-Mannheim/tesseract/wiki")
            else:
                st.subheader("Extracted Text")
                st.text_area("Text from report", result["extracted_text"], height=200)
                
                st.subheader("Health Status Interpretation")
                st.markdown(result["llm_response"])
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Workflow execution failed: {str(e)}")

if __name__ == "__main__":
    main()
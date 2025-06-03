# Medical Test Report Analyzer

## Description

Welcome to the **Medical Test Report Analyzer**, a powerful Python application designed to make medical test reports accessible to everyone. By uploading an image of a medical test report (e.g., blood tests, X-rays), users receive a clear, concise interpretation of their health status, including any abnormal results and actionable recommendations. This tool combines **Optical Character Recognition (OCR)** with advanced AI to extract and analyze text, presenting results in an intuitive [Streamlit](https://streamlit.io/) web interface. Whether you're a non-medical user seeking clarity or a developer exploring AI in healthcare, this project offers a practical and innovative solution.

## Why Use This Project?

- **Simplify Medical Reports**: Get easy-to-understand explanations without medical expertise.
- **AI-Driven Insights**: Leverage advanced language models for accurate health interpretations.
- **User-Friendly Interface**: Upload images and view results effortlessly via Streamlit.
- **Developer-Friendly**: Open-source with clear setup instructions for contributions.
- **Privacy-Focused**: Runs locally with Ollama, ensuring data stays on your device.

## Features

- **Text Extraction**: Uses [Pytesseract](https://github.com/tesseract-ocr/tesseract) for accurate OCR to extract text from image-based reports.
- **AI-Powered Analysis**: Employs [Ollama](https://ollama.com/) with the "qwen2.5:latest" model to generate user-friendly health summaries.
- **Intuitive Interface**: Built with Streamlit, offering seamless image uploads and result displays.
- **Structured Workflow**: Utilizes [LangGraph](https://langchain-ai.github.io/langgraph/) to manage the process from upload to interpretation.
- **Robust Error Handling**: Includes logging and checks for issues like failed OCR or AI processing.
- **Structured Output**: Provides a health status summary, abnormal results, and recommendations.

## Installation

Follow these steps to set up the Medical Test Report Analyzer:

1. **Install Python 3.8 or Higher**:
   - Download from [Python.org](https://www.python.org/downloads/).
   - Verify installation:
     ```bash
     python --version
     ```

2. **Install Required Libraries**:
   - Install dependencies using pip:
     ```bash
     pip install streamlit pillow langgraph langchain-core langchain-ollama pytesseract
     ```

3. **Install Tesseract OCR**:
   - **Windows**: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to your system PATH.
   - **Linux/Mac**: Install via package manager:
     ```bash
     # Ubuntu/Debian
     sudo apt-get install tesseract-ocr
     # macOS (with Homebrew)
     brew install tesseract
     ```

4. **Set up Ollama**:
   - Install from [Ollama.com](https://ollama.com/).
   - Pull the required model:
     ```bash
     ollama pull qwen2.5
     ```
   - Start Ollama:
     ```bash
     ollama serve
     ```

5. **Clone the Repository**:
   - ```bash
     git clone https://github.com/armanjscript/Medical-Test-Report-Analyzer.git
     ```

6. **Navigate to the Project Directory**:
   - ```bash
     cd Medical-Test-Report-Analyzer
     ```

7. **Run the Application**:
   - ```bash
     streamlit run main.py
     ```

**Note**: Ensure sufficient computational resources (e.g., 16GB RAM, CPU/GPU) for running the AI model. An internet connection is required for initial setup.

## Usage

1. **Launch the Application**:
   - Run `streamlit run main.py` to open the app in your default web browser.

2. **Upload an Image**:
   - Click the "Upload Image" button and select a medical test report image (PNG, JPG, JPEG).

3. **View Results**:
   - The app displays:
     - Extracted text from the image.
     - An AI-generated interpretation, including:
       - Health status summary.
       - Abnormal results (if any).
       - Recommendations for next steps.

**Example**:
- Upload a blood test image with "Hemoglobin: 12.5 g/dL, WBC: 7.2 x10^9/L".
- The app extracts the text and provides: "Your hemoglobin is normal, but your WBC count is slightly elevated. Consult a doctor for further evaluation."

## Technologies Used

| Technology       | Role                                                                 |
|------------------|----------------------------------------------------------------------|
| **Python**       | Primary programming language.                                        |
| **Streamlit**    | Creates the interactive web interface.                               |
| **PIL (Pillow)** | Handles image processing and uploads.                                |
| **LangGraph**    | Manages the workflow from upload to analysis.                         |
| **LangChain**    | Integrates with the language model and structures prompts.            |
| **OllamaLLM**    | Runs the "qwen2.5:latest" model for medical interpretations.          |
| **Pytesseract**  | Performs OCR to extract text from images.                             |
| **Logging**      | Tracks operations and aids debugging.                                |
| **Regular Expressions** | Cleans and processes extracted text.                           |

## Contributing

Contributions are encouraged! To contribute:
- Fork the repository on [GitHub](https://github.com).
- Create a new branch for your changes.
- Submit a pull request with a clear description.
- For bug reports or feature requests, open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, open an issue on [GitHub](https://github.com/armanjscript) or [Email] (armannew73@gmail.com).
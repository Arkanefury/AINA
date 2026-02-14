# AINA: Intelligent Personal Assistant (INFOMATRIX 2026)

## 1. Project Overview
*AINA* is an intelligent conversational agent developed as part of a scientific research project for the **INFOMATRIX 2026** competition. The project focuses on creating a personalized, human-like virtual assistant. Unlike standard chatbots, AINA is fine-tuned on a custom dataset to mimic specific linguistic patterns and provide high-context responses.

## 2. Scientific & Technical Features
* Architecture: Built on the `GPT2LMHeadModel` using the `transformers` library.
* Fine-Tuning: The model was trained using the **OpenAINAG** weights, optimized for instruction-following tasks.
* Prompt Engineering: Uses a structured template: `### Instruction: {query} ### Response:` to ensure logical consistency.
* Multimodal Interaction: Supports text and voice input (Web Speech API) and provides synthesized speech output for improved accessibility.
* Localized Context: Specially adapted for the Russian language and personalized dialogue structures.

## 3. Project Structure
* app/main.py: FastAPI server managing the inference pipeline and API endpoints.
* app/model.py: Core model logic, including the `generate_text` function with parameters like top-k (50), top-p (0.95), and temperature control.
* static/index.html: A minimalist, dark-themed UI designed for low latency and high user engagement.
* formatter.py: A data processing script designed to convert raw JSON chat logs into the scientific "Instruction-Response" dataset format.

## 4. Installation & Deployment

### Prerequisites
* Python 3.9+
* NVIDIA GPU (optional but recommended for faster inference)
* Pre-trained weights in `weights/OpenAINAG/`

### Setup

**Clone and Install**: pip install -r requirements.txt

Launch the Application: python run.py

Access the Interface: Navigate to http://localhost:8000 in your browser.

## Privacy and Confidentiality Notice

Please note that the original model weights and the specific datasets used for fine-tuning (derived from Mansur's personal communication logs) are **not included** in this repository and **will not be made publicly available**.

* Confidentiality: The training data contains private dialogues and sensitive information protected under privacy standards.
* Security: To prevent unauthorized reconstruction of personal data, the final fine-tuned weights are restricted to private use only.
* Reproducibility: Researchers wishing to replicate the results are encouraged to use their own datasets with the provided `formatter.py` and the base GPT-2 architecture as described in the methodology.





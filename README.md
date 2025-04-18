# 🧠 From Pixels to Knowledge: Vector Search & Knowledge Graph Workshop

Welcome to **From Pixels to Knowledge**, an AI-powered workshop where we turn raw images into structured data and searchable knowledge graphs using the latest in multimodal AI and graph technologies.

In this workshop, you'll build an interactive app using Streamlit that:

- Extracts structured data from car images using **Gemini API**
- Converts images and text to **vector embeddings** using **CLIP**
- Stores and queries data in **Memgraph** as a knowledge graph
- Supports **vector search** and **text-based querying** with NLP-powered answers


## 🔧 Requirements

- Python 3.9 or above
- Docker & Docker compose
> We recommend starting Docker Compose before the workshop, once you have a stable internet connection.

## 🧰 Install Dependencies

#### Step 1: Create a Virtual Environment in your project root
```bash
cd myWorkspaceProjectRoot
mkdir fp2k && cd fp2k
python -m venv .venv
```
> This will create a `.venv` directory in your project root.
#### Step 2: Activate the Virtual Environment
- On **Linux/macOS**:
```bash
source .venv/bin/activate
```
- On **Windows**:
```bash
.venv\Scripts\activate
```
> Once activated, your shell prompt should show `(.venv)` in front.
#### Step 3: Install Dependencies
Install all Python dependencies using pip:
```bash
pip install -r requirements.txt
```
> Make sure you completed the previous two steps so that all packages are installed inside the virtual environment, keeping your global Python environment clean and avoiding version conflicts.

## 🔑 Environment Variables
Create a .env file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key
```
Make sure to enable Generative AI access in your Google Cloud project and create an API key.
> Google Cloud projects are free for developers, you don't have to worry about billing.

## 🚀 Running the App
Start Memgraph and Memgraph lab using docker compose
```bash
pull 
docker compose up -d
``` 

Start the Streamlit App:
```bash
streamlit run app.py
```

## 🧠 Workshop Features

### 1. Image-Based Search
- Upload an image of a car
- Extract car info with Gemini
- Convert to vector using CLIP
- Search similar cars in Memgraph

### 2. Text-Based Search
- Input a textual prompt
- Convert it to a vector
- Search Memgraph and visualize results
- Let Gemini answer your question with the help of graph data

### 3. Bulk Image Processing
- Upload up to 15 images at once
- Automatically extract metadata
- Store them in Memgraph for later querying

## 🗂 Folder Structure

```bash
.
├── images/                # Images for the demo
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .env                   # API keys and secrets
├── docker-compose.yml     # Docker compose file for memgraph
└── README.md              # You're here!
```

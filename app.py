import base64
import json
import mimetypes
import os
from io import BytesIO

import streamlit as st
import torch
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from neo4j import GraphDatabase
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

torch.classes.__path__ = []


load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
db = GraphDatabase.driver("bolt://localhost:7687", auth=("", ""))


def process_image_with_gemini(image):
    """
    The function uses the Gemini API to analyze the image and extract various features
    such as car model, year, dimensions, description, color, body type, and manufacturer.
    It returns the extracted data in a structured JSON format.
    """

    try:
        mime_type, _ = mimetypes.guess_type(image.name)
        image_bytes = image.read()

        model_name = "gemini-2.0-flash"
        contents = [
            Content(
                role="user",
                parts=[
                    Part.from_bytes(
                        mime_type=mime_type,
                        data=image_bytes,
                    ),
                ],
            ),
        ]

        json_schema = {
            "car": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["model", "year", "dimensions", "description"],
                properties={
                    "model": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The specific model of the car, e.g., Mustang GT",
                    ),
                    "year": genai.types.Schema(
                        type=genai.types.Type.INTEGER,
                        description="The year the car was produced, e.g., 2020",
                    ),
                    "dimensions": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Relative size of the car, e.g., Small, Medium, Large",
                    ),
                    "description": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="Describe the the car, e.g. build quality, comfort, overall looks, features",
                    ),
                },
            ),
            "color": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="The color of the car, e.g., Red, Blue, Black",
            ),
            "body_type": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="The body type of the car, e.g., Sedan, Coupe, SUV",
            ),
            "manufacturer": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["name", "country", "founded_year"],
                properties={
                    "name": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The name of the car manufacturer, e.g., Ford",
                    ),
                    "country": genai.types.Schema(
                        type=genai.types.Type.STRING,
                        description="The country where the manufacturer is based, e.g., USA",
                    ),
                    "founded_year": genai.types.Schema(
                        type=genai.types.Type.INTEGER,
                        description="The year the manufacturer was founded, e.g., 1903",
                    ),
                },
            ),
        }

        generate_content_config = GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            top_k=64,
            response_mime_type="application/json",
            response_schema=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=["car", "color", "body_type", "manufacturer"],
                properties=json_schema,
            ),
            system_instruction=[
                Part.from_text(
                    text=f"""Analyze the provided car image and extract following visual features:
                    {json_schema}
                    Provide the extracted information in a structured JSON format adhering to the provided schema. All JSON fields are mandatory.
                    """
                ),
            ],
        )

        response = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response += chunk.text

        return json.loads(response)

    except json.JSONDecodeError:
        st.error(
            "Received malformed JSON from Gemini API. Check the API response format."
        )
        return {"error": "Invalid JSON response from API"}

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}


def convert_image_to_vector(image):
    """
    Converts an image to a feature vector using Hugging Face CLIP.
    """
    image = Image.open(image).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Get image embeddings
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Normalize the vector
    image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy()


def encode_image_to_base64(image):
    """
    Encodes an image file to a base64 string.
    """
    image.seek(0)
    return base64.b64encode(image.read()).decode("utf-8")


def construct_cypher_queries(extracted_data, embeddings, image_base64):
    """
    Constructs a Cypher query from the extracted JSON data.
    """
    car = extracted_data.get("car", {})
    color = extracted_data.get("color", {})
    body_type = extracted_data.get("body_type", {})
    manufacturer = extracted_data.get("manufacturer", {})
    queries = []

    # Query to create Manufacturer node
    manufacturer_query = f"""
    MERGE (m:Manufacturer {{name: '{manufacturer.get("name")}', 
                            country: '{manufacturer.get("country")}', 
                            founded_year: {manufacturer.get("founded_year")}}})
    RETURN m;
    """
    queries.append(manufacturer_query)

    # Query to create Car node
    car_query = f"""
    MERGE (c:Car {{make: '{car.get("make")}', 
                   model: '{car.get("model")}', 
                   year: {car.get("year")}, 
                   dimensions: '{car.get("dimensions")}', 
                   description: '{car.get("description")}',
                   embedding: {embeddings[0]},
                   image: '{image_base64}'}})
    RETURN c;
    """
    queries.append(car_query)

    # Add color node
    color_query = f"""
    MERGE (c:Color {{name: '{color}'}})
    """
    queries.append(color_query)

    # Add body type node
    body_type_query = f"""
    MERGE (b:BodyType {{name: '{body_type}'}})
    """
    queries.append(body_type_query)

    # Query to create the relationship between Car and Manufacturer
    relationship_query = f"""
    MATCH (c:Car {{make: '{car.get("make")}', model: '{car.get("model")}', year: {car.get("year")}}})
    MATCH (m:Manufacturer {{name: '{manufacturer.get("name")}'}})
    MERGE (c)-[:MADE_BY]->(m)
    RETURN c, m;
    """
    queries.append(relationship_query)

    # Query to create the relationship between Car and Color
    color_relationship_query = f"""
    MATCH (c:Car {{make: '{car.get("make")}', model: '{car.get("model")}', year: {car.get("year")}}})
    MATCH (col:Color {{name: '{color}'}})
    MERGE (c)-[:HAS_COLOR]->(col)
    RETURN c, col;
    """
    queries.append(color_relationship_query)

    # Query to create the relationship between Car and BodyType
    body_type_relationship_query = f"""
    MATCH (c:Car {{make: '{car.get("make")}', model: '{car.get("model")}', year: {car.get("year")}}})
    MATCH (b:BodyType {{name: '{body_type}'}})
    MERGE (c)-[:HAS_BODY_TYPE]->(b)
    RETURN c, b;
    """
    queries.append(body_type_relationship_query)

    return queries


def run_memgraph_query(query):
    """
    Executes a Cypher query on the Memgraph database and returns the results.
    """
    try:
        with db.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return {"error": str(e)}


def decode_image_to_base64(image_base64):
    """
    Decodes a base64 image string into an image.
    """
    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes))


def answer_NLP_query(query: str, data: str) -> str:
    """
    Uses the Gemini API to answer a natural language question based on the given text data.

    Args:
        query (str): The question to ask.
        data (str): The text context to use as reference for answering.

    Returns:
        str: The answer generated by the model.
    """
    try:
        model_name = "models/gemini-2.0-flash"

        contents = [
            Content(
                role="user",
                parts=[
                    Part.from_text(
                        text="Answer the question based on the provided data."
                    ),
                    Part.from_text(text=f"Question: {query}"),
                    Part.from_text(text=f"Data: {data}"),
                ],
            )
        ]

        generate_content_config = GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            top_k=64,
            response_mime_type="text/plain",
        )

        response = ""
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response += chunk.text

        return response.strip()

    except Exception as e:
        return f"Error: {e}"


def setup_indexes():
    """
    Sets up indexes in the Memgraph database for efficient querying.
    Ensures the index is created only once.
    """
    try:
        with db.session() as session:
            result = session.run("SHOW INDEX INFO;")
            existing_indexes = [
                (record["label"], record["property"]) for record in result
            ]

            if ("Car", "embedding") not in existing_indexes:
                session.run(
                    """CREATE VECTOR INDEX car_embedding ON :Car(embedding) WITH CONFIG {"dimension": 512, "capacity": 1000};"""
                )
                print("Index 'car_embedding' created.")
            else:
                print("Index 'car_embedding' already exists.")

    except Exception as e:
        st.error(f"Error setting up indexes: {e}")
        return {"error": str(e)}


def perform_memgraph_vector_search(index, query_vector):
    """
    Performs a vector search in Memgraph using the provided query vector.
    """
    query_vector = query_vector.tolist()
    query = f"""
    CALL vector_search.search("{index}", 3, {query_vector[0]}) YIELD * RETURN *;
    """
    try:
        with db.session() as session:
            result = session.run(query)
            memgraph_response = [record.data() for record in result]
            return memgraph_response
    except Exception as e:
        st.error(f"Error executing vector search: {e}")
        return {"error": str(e)}


def convert_text_to_vector(text: str):
    """
    Converts a text string to a feature vector using Hugging Face CLIP.
    """
    # Preprocess the text
    inputs = processor(text=[text], return_tensors="pt").to(device)

    # Get text embeddings
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # Normalize the vector
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


def strip_base64_from_results(results):
    """
    Removes the 'image' and 'embedding' fields from Memgraph result records for keys 'node' and 'c'
    to improve readability in the UI.
    """
    keys_to_check = ["node", "c"]
    stripped_results = []

    for record in results:
        cleaned_record = {}
        for key, value in record.items():
            if key in keys_to_check and isinstance(value, dict):
                # Remove the "image" and "embedding" fields
                cleaned_node = {
                    k: v for k, v in value.items() if k not in ["image", "embedding"]
                }
                cleaned_record[key] = cleaned_node
            else:
                cleaned_record[key] = value
        stripped_results.append(cleaned_record)

    return stripped_results


def get_relevant_data(db_client, node, hops):
    """
    Retrieves relevant data paths from the Memgraph database for a given node and number of hops.
    Ensures 'embedding' and 'image' properties are excluded from the node data.
    """
    paths = []
    try:
        description = node.get("description")
        with db_client.session() as session:
            query = f'MATCH path=((n:Car {{description: "{description}"}})-[r*..{hops}]-(m)) RETURN path'
            result = session.run(query)

            for record in result:
                path_data = []
                for segment in record["path"]:
                    # Process start node without 'embedding' and 'image' properties
                    start_node_data = {
                        k: v
                        for k, v in segment.start_node.items()
                        if k not in ["embedding", "image"]
                    }

                    # Process relationship data
                    relationship_data = {
                        "type": segment.type,
                        "properties": segment.get("properties", {}),
                    }

                    # Process end node without 'embedding' and 'image' properties
                    end_node_data = {
                        k: v
                        for k, v in segment.end_node.items()
                        if k not in ["embedding", "image"]
                    }

                    # Add to path_data as a tuple (start_node, relationship, end_node)
                    path_data.append(
                        (start_node_data, relationship_data, end_node_data)
                    )

                paths.append(path_data)
    except Exception as e:
        st.error(f"Error retrieving relevant data: {e}")
        return {"error": str(e)}

    return paths


def app():
    st.title("Vector Search - Memgraph Knowledge Graph Workshop")

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Image Query", "💬 Text Query", "🛠️ Bulk Image Processing & Storage"]
    )

    # --- TAB 3: BULK IMAGE PROCESSING ---
    with tab3:
        st.subheader(
            "Upload up to 15 images to extract metadata and store into Memgraph"
        )
        uploaded_images = st.file_uploader(
            "Upload images...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="multi_image_upload",
        )

        if uploaded_images:
            if len(uploaded_images) > 15:
                st.warning("Please upload no more than 15 images.")
                uploaded_images = uploaded_images[:15]

            for idx, uploaded_image in enumerate(uploaded_images):
                st.divider()
                st.markdown(f"### 🖼️ Image {idx + 1}")
                st.image(
                    uploaded_image,
                    caption=f"Uploaded Image {idx + 1}",
                    use_container_width=True,
                )

                # Step 1: Gemini Extraction
                with st.spinner("Processing image with Gemini..."):
                    extracted_data = process_image_with_gemini(uploaded_image)
                st.subheader("Extracted Data")
                st.json(extracted_data)

                # Step 2: Convert to Vector
                with st.spinner("Converting image to Vector..."):
                    vector = convert_image_to_vector(uploaded_image)
                st.write(f"Vector (partial): {vector[0][:10]}...")

                # Step 3: Generate and show Cypher
                image_base64 = encode_image_to_base64(uploaded_image)
                cypher_queries = construct_cypher_queries(
                    extracted_data=extracted_data,
                    embeddings=vector.tolist(),
                    image_base64=image_base64,
                )

                st.subheader("Cypher Queries")
                for query in cypher_queries:
                    st.code(query)

                # Step 4: Execute Cypher
                st.subheader("Query Results")
                with st.spinner("Executing Cypher queries..."):
                    for query in cypher_queries:
                        result = run_memgraph_query(query)
                        clean_results = strip_base64_from_results(result)
                        st.json(clean_results)
        else:
            st.info("Please upload image files to extract and store information.")

    # --- TAB 1: IMAGE QUERY ---
    with tab1:
        st.subheader("Upload an image for querying")
        query_image = st.file_uploader(
            "Upload image...", type=["jpg", "jpeg", "png"], key="image_query_upload"
        )

        if query_image:
            st.image(query_image, caption="Uploaded Image", use_container_width=True)

            st.subheader("Convert Image to Vector")
            with st.spinner("Converting image to Vector..."):
                vector_image = convert_image_to_vector(query_image)
            st.write(f"Vector (partial): {vector_image[0][:10]}...")

            st.subheader("Perform vector search in Memgraph")
            with st.spinner("Performing vector search..."):
                results = perform_memgraph_vector_search(
                    index="car_embedding", query_vector=vector_image
                )
                st.success("Memgraph Results:")
                clean_results = strip_base64_from_results(results)
                st.json(clean_results)

                st.subheader("Top Result Image")
                if results:
                    top_result = results[0]
                    image_base64 = top_result.get("node").get("image")
                    image = decode_image_to_base64(image_base64)
                    st.image(
                        image, caption="Top Result Image", use_container_width=True
                    )
        else:
            st.info("Please upload an image to perform the image query.")

    # --- TAB 2: TEXT QUERY ---
    with tab2:
        st.subheader("Enter a text prompt for querying")
        text_query = st.text_input("Text Query", key="text_query_input")
        query_button = st.button("Run Text Query", key="text_query_button")

        if query_button and text_query:
            st.subheader(f"Convert text: '{text_query}' to Vector")
            with st.spinner("Converting text to Vector..."):
                vector_query = convert_text_to_vector(text_query)
            st.write(f"Vector (partial): {vector_query[0][:10]}...")

            st.subheader("Perform vector search in Memgraph")
            with st.spinner("Performing vector search..."):
                results = perform_memgraph_vector_search(
                    index="car_embedding", query_vector=vector_query
                )

                st.success("Memgraph Results:")
                clean_results = strip_base64_from_results(results)
                st.json(clean_results)
                top_result = results[0] if results else None

            st.subheader("Getting relevant data from top node Memgraph")
            if results:
                image_base64 = top_result.get("node").get("image")
                image = decode_image_to_base64(image_base64)
                st.image(image, caption="Top Result Image", use_container_width=True)
                st.subheader("Relevant Data")
                relevant_data = get_relevant_data(
                    db, node=top_result.get("node"), hops=2
                )
                st.json(relevant_data)

                st.subheader("Answer the question using Gemini")
                with st.spinner("Answering question..."):
                    answer = answer_NLP_query(
                        query=text_query,
                        data=relevant_data,
                    )
                st.success("Gemini Answer:")
                st.text(answer)
        else:
            st.info("Please enter a text query and click the button.")


if __name__ == "__main__":
    setup_indexes()
    app()

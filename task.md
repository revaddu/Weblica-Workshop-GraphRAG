# Enhance Image Upload Pipeline

Try to detect the location of vehicle! 

## Overview
This task expands the existing image upload pipeline. The pipeline will:
- **Extract data and Determine location:** Identify license plate details or environmental cues in the image.
- **Update graph database:** Create or update corresponding `:City` or `:Country` nodes and link the `:Car` node via a `:REGISTERED` relationship.
- **Query functionality:** Fetch cars based on location after uploading several images.

## Pipeline Details

### Step 1: Image Processing
- **Input:** Uploaded image.
- **Action:** Call Gemini to help you decide on the image data based on: 
  - License plate information
  - Environmental clues indicating location

### Step 2: Graph Database Update
- **Operations:**
  - Create or update the graph node: either `:City` or `:Country`.
  - Establish the `:REGISTERED` relationship between the `:Car` node and the location node.

### Step 3: Query Cars by Location
- **Post-process:** Retrieve cars tied to a specific location based on the developed relationships.

### Step 4: Ask questions
- **Ask questions:** Ask pipeline the NLP questions about location of vehicles 
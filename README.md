# Research Paper Assistant — RAG System

## Project Overview

The goal of this project is to design and implement an end-to-end data flow for a Retrieval-Augmented Generation (RAG) system that can retrieve information from scientific papers and use it to answer research-related questions. When the full text processing system will be ready, an image interpretation will be possible added, if time allows.

The project focuses on building the complete pipeline: collecting scientific papers, extracting and preprocessing their text, splitting documents into chunks, storing them in a vector database, retrieving relevant information, and providing the retrieved context to a local Large Language Model (LLM).

## Data Source

The dataset consists of approximately 100 recent research papers from the **Artificial Intelligence (cs.AI)** category on arXiv.

The arXiv API is used to collect paper metadata and download the corresponding PDF files. The collected information includes:

- Article ID
- Title
- Abstract
- Publication date
- arXiv URL
- PDF document

Images are extracted from the downloaded PDF files for further processing.

## Data Processing and EDA

The extracted article text is divided into fixed-size overlapping chunks. Each chunk contains its article ID, title, chunk index, and text.

Exploratory Data Analysis is performed to examine the processed dataset and verify that the documents were transformed correctly.

The analysis includes:

- Missing-value and duplicate checks
- Analysis of the number of chunks per article
- Analysis of the distribution of article chunk counts
- Identification of potential outliers
- Word-frequency analysis
- Stop-word analysis

The analysis showed differences in article length and therefore in the number of generated chunks. Word-frequency analysis also identified stop words and mathematical symbols that may introduce noise into the textual data.

## RAG Pipeline

The implemented pipeline follows these main steps:

**arXiv → PDF extraction → text preprocessing → chunking → vector database → aprox nearest neighbor → context augmentation → local LLM → generated answer**

For this prototype I was using only ANN. algo for searching, and for the small amount data it worked very well. However, I am planning to build a better evaluation workflow and perform a hybrid search for the documtns.
### Vector Database

**Weaviate** is used as the knowledge base and vector database.

Processed article chunks are uploaded to Weaviate and retrieved using vector similarity search based on the user's question.

### Language Model

The generation component uses a locally hosted **Qwen3-8B** model.

The retrieved document chunks are added to the model's context. The model is instructed to answer using only the retrieved information and to indicate when the available context is insufficient.

## Retrieval Evaluation

The retrieval component is evaluated using **Mean Average Precision (MAP)** and **Recall@K**.

**Recall@K** measures whether the expected relevant article appears among the top K retrieved results.

**MAP** takes the ranking of the relevant result into account, giving a higher score when the expected article appears closer to the top of the retrieved results.

The current evaluation produced strong retrieval results. However, these results should be interpreted with caution because the evaluation dataset is relatively small and the test questions are semantically related to the source articles.

A larger and more diverse evaluation dataset would provide a more robust assessment of retrieval performance.

## Limitations and Future Work

The primary objective of this project was to build a functional end-to-end data flow for a RAG system rather than a traditional classification or regression model. 

A more comprehensive evaluation of generated answers using **RAGAS** was explored. However, generating a synthetic RAGAS evaluation dataset requires multiple LLM and embedding operations and proved computationally expensive and time-consuming when using locally hosted models.

Future improvements include:

- RAGAS evaluation of answer faithfulness and relevancy
- A larger retrieval evaluation dataset
- Experimentation with different chunk sizes and overlap
- Improved document selection and retrieval strategies
- Comparison of different embedding models
- Potential integration of figures and images extracted from research papers

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- NLTK
- PyMuPDF
- Weaviate
- Hugging Face Transformers
- Qwen3-8B
- arXiv API

## Jupyter Notebook

The complete data analysis, RAG implementation, retrieval evaluation, and visualizations are available in the project Jupyter notebook:

**[Research Paper Assistant Notebook](https://github.com/vilyapilya/scientific_article_RAG/blob/main/arxiv_research_assistant_eda.ipynb)**



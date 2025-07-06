# 🧠 Biomedical Knowledge Graph Embedding Pipeline for Disease Susceptibility Analysis

This repository provides a modular pipeline for constructing and embedding a biomedical knowledge graph that integrates heterogeneous biological entities and relationships. It enables advanced representation learning across diseases, genes, chemicals, and gene ontology terms to support graph-based inference in biomedical domains.

### Included Components

- **`data_preprocessing.py`**  
  Processes raw datasets containing disease-gene, disease-GO, and disease-chemical associations. Converts them into structured (head, relation, tail) triples and constructs a fully labeled knowledge graph in Neo4j.

- **`embedding_generator.py`**  
  Builds vocabularies for all entities and relations, initializes dense embeddings using Xavier initialization, and saves them in a PyTorch-compatible format for downstream training.

- **`gene-extraction-pipeline.py`**  
  Automates extraction of gene mentions from biomedical literature using named entity recognition and prepares data for integration into the knowledge graph.

- **`kg_cnn_model.py`**  
  Implements a CNN-based model architecture for learning from knowledge graph embeddings using multi-scale convolutional filters and training logic for link prediction and association scoring.

### Final Outcome

The complete pipeline enables the computation of a **disease susceptibility index**, capturing complex interactions between diseases, genes, chemicals, and biological processes.

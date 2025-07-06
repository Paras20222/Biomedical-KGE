# data_preprocessing_and_neo4j.py

import os
import pandas as pd
from neo4j import GraphDatabase

# --- Config ---
DATA_DIR = "./raw_data"  # Directory containing CSVs
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# --- Triple Schema Mapping ---
SCHEMA = {
    "genes.csv": {
        "head_col": "DiseaseID", "tail_col": "GeneID",
        "head_type": "Disease", "tail_type": "Gene", "relation": "INVOLVES"
    },
    "go_terms.csv": {
        "head_col": "DiseaseID", "tail_col": "GO_ID",
        "head_type": "Disease", "tail_type": "GO", "relation": "RELATED_TO"
    },
    "chemicals.csv": {
        "head_col": "DiseaseID", "tail_col": "ChemicalID",
        "head_type": "Disease", "tail_type": "Chemical", "relation": "TREATED_BY"
    }
}

# --- Neo4j Graph Loader ---
class GraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_graph(self, triples_df):
        with self.driver.session() as session:
            for _, row in triples_df.iterrows():
                session.write_transaction(
                    self._create_relation,
                    row['head'], row['relation'], row['tail'],
                    row['head_type'], row['tail_type']
                )

    @staticmethod
    def _create_relation(tx, head, rel, tail, head_type, tail_type):
        query = (
            f"MERGE (h:{head_type} {{id: $head}}) "
            f"MERGE (t:{tail_type} {{id: $tail}}) "
            f"MERGE (h)-[r:{rel}]->(t)"
        )
        tx.run(query, head=head, tail=tail)

# --- Triple Extractor ---
def load_all_triples(data_dir, schema):
    all_triples = []
    for file, config in schema.items():
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)

        df = df[[config["head_col"], config["tail_col"]]].dropna()
        df.columns = ["head", "tail"]
        df["relation"] = config["relation"]
        df["head_type"] = config["head_type"]
        df["tail_type"] = config["tail_type"]

        all_triples.append(df)
    return pd.concat(all_triples, ignore_index=True)

if __name__ == "__main__":
    triples_df = load_all_triples(DATA_DIR, SCHEMA)
    triples_df.to_csv("triples.csv", index=False)

    builder = GraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    builder.create_graph(triples_df)
    builder.close()

import polars as pl
import chromadb
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import shutil

# Import NCM data and concatenate info
df = pl.read_csv('data/BaseDESC_NCM.csv', separator='\t', encoding='utf-8', n_threads=18)

# Process rows to create documents and metadata
def process_row(row):
    document = f"NCM: {row['NCM']}, Rótulo: {row['rotulo']}, Item: {row['Item']}, Produto: {row['XPROD']}"
    metadata = {
        "NCM": row['NCM'],
        "rotulo": row['rotulo'],
        "Item": row['Item'],
        "Produto": row['XPROD']
    }
    return document, metadata

def sanitize_metadata(metadata):
    sanitized = {}
    for key, value in metadata.items():
        # Ensure keys are strings and values are strings, numbers, or booleans
        if isinstance(key, str):
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)  # Convert to string if not a basic type
    return sanitized

# Parallel process
with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_row, df.to_dicts()), total=len(df), desc="Processing rows"))

documents, metadatas = zip(*results)

# Load a pre-trained Portuguese sentence transformer model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(documents)

# Create persistent chromadb
try:
    shutil.rmtree("chroma_db")
    print("Successfully removed chroma_db directory")
except FileNotFoundError:
    print("chroma_db directory not found, so it could not be removed")
except Exception as e:
    print(f"An error occurred while removing chroma_db: {e}")

# Create persistent chromadb
print("Start persistent chromadb")
persist_directory = "chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection_name = "ncm-all-data"
collection = client.get_or_create_collection(name=collection_name)

# Populate chromadb in batches
batch_size = 41666  # Adjust batch size as needed
print("Start populating chromadb")
for i in tqdm(range(0, len(documents), batch_size), desc="Adding batches to ChromaDB"):
    batch_end = min(i + batch_size, len(documents))
    batch_documents = documents[i:batch_end]
    batch_embeddings = embeddings[i:batch_end]
    batch_metadatas = [sanitize_metadata(metadata) for metadata in metadatas[i:batch_end]]
    batch_ids = [f"id{j}" for j in range(i, batch_end)]

    collection.add(
        documents=batch_documents,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas,
        ids=batch_ids
    )

# client.persist() # Persist the data
# print("ChromaDB persistence completed.")
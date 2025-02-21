import polars as pl
import chromadb
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import shutil

# Import NCM data and concatenate info
with tqdm(total=100, desc="Carregando dados") as pbar:
    df = pl.read_csv('data/BaseDESC_NCM.csv', separator='\t', encoding='utf-8', n_threads=18)
    pbar.update(100)

# Process rows to create documents and metadata
def process_row(row):
    # document = f"NCM: {row['NCM']}, Rótulo: {row['rotulo']}, Item: {row['Item']}, Produto: {row['XPROD']}"
    # return document
    return row

# Parallel process
with ThreadPoolExecutor() as executor:
    documents = list(tqdm(executor.map(process_row, df.to_dicts()), total=len(df), desc="Processando linhas"))

print(documents[0:5])

# Load a pre-trained Portuguese sentence transformer model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Create embeddings for each field separately
ncm_texts = [str(doc['NCM']) for doc in documents]
rotulo_texts = [str(doc['rotulo']) for doc in documents]
item_texts = [str(doc['Item']) for doc in documents]
xprod_texts = [str(doc['XPROD']) for doc in documents]

ncm_embeddings = model.encode(ncm_texts, batch_size=512, convert_to_tensor=True, show_progress_bar=True).tolist()
rotulo_embeddings = model.encode(rotulo_texts, batch_size=512, convert_to_tensor=True, show_progress_bar=True).tolist()
item_embeddings = model.encode(item_texts, batch_size=512, convert_to_tensor=True, show_progress_bar=True).tolist()
xprod_embeddings = model.encode(xprod_texts, batch_size=512, convert_to_tensor=True, show_progress_bar=True).tolist()


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
batch_size = 41666  # Max batch size
print("Start populating chromadb")
with tqdm(total=len(documents), desc="Adding to ChromaDB") as pbar:
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        batch_documents = documents[i:batch_end]

        batch_ncm_embeddings = ncm_embeddings[i:batch_end]
        batch_rotulo_embeddings = rotulo_embeddings[i:batch_end]
        batch_item_embeddings = item_embeddings[i:batch_end]
        batch_xprod_embeddings = xprod_embeddings[i:batch_end]

        metadatas = [{
            "NCM": doc['NCM'],
            "rotulo": doc['rotulo'],
            "Item": doc['Item'],
            "Produto": doc['XPROD']
        } for doc in batch_documents]

        collection.add(
            embeddings=batch_ncm_embeddings,
            metadatas=metadatas,
            ids=[f"ncm_id{j}" for j in range(i, batch_end)],
            documents = [f"NCM: {doc['NCM']}" for doc in batch_documents]
        )
        collection.add(
            embeddings=batch_rotulo_embeddings,
            metadatas=metadatas,
            ids=[f"rotulo_id{j}" for j in range(i, batch_end)],
            documents = [f"Rótulo: {doc['rotulo']}" for doc in batch_documents]
        )
        collection.add(
            embeddings=batch_item_embeddings,
            metadatas=metadatas,
            ids=[f"item_id{j}" for j in range(i, batch_end)],
            documents = [f"Item: {doc['Item']}" for doc in batch_documents]
        )
        collection.add(
            embeddings=batch_xprod_embeddings,
            metadatas=metadatas,
            ids=[f"xprod_id{j}" for j in range(i, batch_end)],
            documents = [f"Produto: {doc['XPROD']}" for doc in batch_documents]
        )
        pbar.update(len(batch_documents))

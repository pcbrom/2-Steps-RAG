import polars as pl
import chromadb
# from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# Import NCM data and concatenate info
df = pl.read_csv('data/BaseDESC_NCM.csv', separator='\t', encoding='utf-8', n_threads=18)

# Collapse text
def process_row(row):
    text = f"NCM: {row['NCM']}, Rótulo: {row['rotulo']}, Item: {row['Item']}, Produto: {row['XPROD']}"
    return text

# Parallel process
with ThreadPoolExecutor() as executor:
    documents = list(tqdm(executor.map(process_row, df.to_dicts()), total=len(df), desc="Processing rows"))

# Load a pre-trained Portuguese sentence transformer model
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(documents[0:5])

# Create persistent chromadb
persist_directory = "chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection_name = "ncm-all-data"
collection = client.get_or_create_collection(name=collection_name)

# 
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"id{i}" for i in range(len(df))]
)
import pandas as pd
import chromadb
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# Import NCM data and concatenate info
df = pd.read_csv('data/BaseDESC_NCM.csv', sep='\t', encoding='utf-8')

def process_row(row):
    text = f"NCM: {row['NCM']}, Rótulo: {row['rotulo']}, Item: {row['Item']}, Produto: {row['XPROD']}"
    return text

with ThreadPoolExecutor() as executor:
    documents = list(tqdm(executor.map(process_row, df.to_dict('records')), total=len(df), desc="Processing rows"))
print(documents[0:10])

# Create persistent chromadb
persist_directory = "chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection_name = "ncm-all-data"
collection = client.get_or_create_collection(name=collection_name)

# 

"""collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(df))]
)
"""
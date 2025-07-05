import warnings
warnings.filterwarnings('ignore')

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from DLAIUtils import Utils
# import DLAIUtils

import os
import time
import torch

from tqdm.auto import tqdm

dataset = load_dataset('quora', split='train[240000:290000]')

questions = []

for record in dataset['questions']:
    questions.extend(record["text"])
question = list(set(questions))

print('\n'.join(questions[:10]))
print('-'*50)
print(f'number of quesions : {len(questions)}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device.........................{device}')
if device != 'cuda':
    print("Sorry cuda not available")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

query = 'What is nifty?'
xq= model.encode(query)
xq.shape

utils = Utils()

PINECONE_API_KEY = utils.get_pinecone_api_key()

pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = utils.create_dlai_index_name('dl-ai')

print(INDEX_NAME,'indexname..............')

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
    pinecone.delete_index(INDEX_NAME)
print(INDEX_NAME)

pinecone.create_index(name=INDEX_NAME,dimension=model.get_sentence_embedding_dimension(),metric='cosine',spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pinecone.Index(INDEX_NAME)
print(index)

# Create Embeddings and Upsert to Pincone

batch_size = 200
vector_limit = 10000

questions = question[:vector_limit]

import json 

for i in tqdm(range(0, len(questions),batch_size)):
    #find end of the batch
    i_end = min(i+batch_size, len(questions))

    ids = [str(x) for x in range(i, i_end)]
    
    #metadata creation and preparation
    metadatas = [{"text" : text} for text in questions[i:i_end]]

    #create embeddings
    xc = model.encode(questions[i:i_end])

    #create record for upserting
    records = zip(ids, xc, metadatas)

    # upsert to the pinecone
    index.upsert(vectors=records)

    index.describe_index_stats()

# small helper function so we can repeat queries later
def run_query(query):
    embedding = model.encode(query).tolist()
    results = index.query(top_k=10,vector=embedding, include_metadata=True, include_values=False)
    for result in results['matches']:
        print(f"{round(result['score'],2)}:{result["metadata"]["text"]}")

run_query('which city has the highest population in the world?')



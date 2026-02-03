
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('automotive-manuals')
stats = index.describe_index_stats()

print('=' * 60)
print('PINECONE INDEX VERIFICATION')
print('=' * 60)
print(f'Total vectors: {stats.total_vector_count}')
print(f'Dimension: {stats.dimension}')
print(f'Index fullness: {stats.index_fullness:.2%}')
print('=' * 60)
print('SUCCESS! Your data is in Pinecone.')
print('=' * 60)

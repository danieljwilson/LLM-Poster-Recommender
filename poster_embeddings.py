# Imports
import pandas as pd
import numpy as np
import openai
import os
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

#############
# LOAD DATA #
#############

# Path to datafile
df = pd.read_excel('data/agenda_export_spspa_202402 MJ.xls', sheet_name=0)

################
# PROCESS DATA #
################

# Filter for poster sessions
df_posters = df.loc[df['Tracks'].str.contains('Poster', na=False) & 
                    ~df['*Session Title'].str.contains('Poster Session', na=False) &
                    df['Authors'].notna()].reset_index(drop=True)

# Remove rows with no description
df_posters = df_posters.dropna(subset=['Description'])

# Edit titles to remove the initial "[number]" portion
df_posters['*Session Title'] = [re.sub(r"^\[\d+\]", "", title).strip() for title in df_posters['*Session Title']]

##################
# GET EMBEDDINGS #
##################

# Access the API key from an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

from openai import OpenAI
client = OpenAI()

# Function to get embeddings
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Combine Title and Description
df_posters['Description'] = df_posters['Description'].astype(str)
df_posters['combined'] = df_posters['*Session Title'] + ' ' + df_posters['Description']

# Get embeddings
df_posters['embedding_3small'] = df_posters.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

# Create embeddings matrix
df_posters.embedding_3small.apply(eval).apply(np.array)

for index, row in df_posters.iterrows():
    if index ==0:
        embeddings_array = np.array(df_posters['embedding_3small'][index])
    if index >0:
        embeddings_array = np.vstack((embeddings_array, np.array(df_posters['embedding_3small'][index])))

np.savez('output/embeddings_array.npz', embeddings_array = embeddings_array)

##################
# DIM. REDUCTION #
##################

# PCA
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings_array)

# Add to df
df_posters['pca1'] = embeddings_2d_pca[:,0]
df_posters['pca2'] = embeddings_2d_pca[:,1]

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d_tsne = tsne.fit_transform(embeddings_array)

# Add to df
df_posters['tSNE1'] = embeddings_2d_tsne[:,0]
df_posters['tSNE2'] = embeddings_2d_tsne[:,1]

# Save dataframe
df_posters.to_csv('output/spsp_wEmbeddings.csv', index=False)

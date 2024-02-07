# Imports
import pandas as pd
import numpy as np
import openai
import os
import re
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px

# Load data
df = pd.read_excel('data/agenda_export_spspa_202402 MJ.xls', sheet_name=0)

# Filter for poster sessions
df_posters = df.loc[df['Tracks'].str.contains('Poster', na=False) & 
                    ~df['*Session Title'].str.contains('Poster Session', na=False) &
                    df['Authors'].notna()].reset_index(drop=True)

# Remove rows with no description
df_posters = df_posters.dropna(subset=['Description'])

# Edit titles to remove the initial "[number]" portion
df_posters['*Session Title'] = [re.sub(r"^\[\d+\]", "", title).strip() for title in df_posters['*Session Title']]

# # Save data
# df_posters.to_csv('spsp.csv', index=False)

##################
# GET EMBEDDINGS #
##################

# Access the API key from an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

from openai import OpenAI
client = OpenAI()

df_posters['Description'] = df_posters['Description'].astype(str)

# Function to get embeddings
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Add title to embedding
df_posters['combined'] = df_posters['*Session Title'] + ' ' + df_posters['Description']

# Get embeddings
start_time = time.time() # Capture the start time
df_posters['embedding_3small'] = df_posters.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
end_time = time.time() # Capture the end time
duration = end_time - start_time # Calculate the duration
print(duration)

####################
# Create embeddings matrix
df_posters.embedding_3small.apply(eval).apply(np.array)

for index, row in df_posters.iterrows():
    if index ==0:
        embeddings_array = np.array(df_posters['embedding_3small'][index])
    if index >0:
        embeddings_array = np.vstack((embeddings_array, np.array(df_posters['embedding_3small'][index])))

print("Shape: " + str(embeddings_array.shape))
print(embeddings_array)

np.savez('output/embeddings_array.npz', embeddings_array = embeddings_array)

##########
# Search term comparison
import ast
from sklearn.metrics.pairwise import cosine_similarity

def search_posters(df, embeddings_array, query_term):
    embedding = get_embedding(query_term, model='text-embedding-3-small')
    embedding = np.array(embedding).reshape(1, -1)
    similarities = cosine_similarity(embeddings_array, embedding).flatten()
    df['Similarity'] = similarities
   
   # Get indices for 10 most similar, 5 most different, and 5 random entries
    similar_indices = np.argsort(similarities)[-11:]  # Last 10 plus the selected one itself
    different_indices = np.argsort(similarities)[:5]
    random_indices = np.random.choice(df.index.difference(similar_indices).difference(different_indices), 5, replace=False)
    
    combined_indices = np.unique(np.concatenate((similar_indices, different_indices, random_indices)))
    res = df.loc[combined_indices].copy()
    # Create names of groups
    res['group'] = ''
    res.loc[list(similar_indices), 'group'] = 'Similar'
    res.loc[list(different_indices), 'group'] = 'Different'
    res.loc[list(random_indices), 'group'] = 'Random'
    
    res = res.sort_values('Similarity', ascending=False)
    return res

plot_df = search_posters(df_posters, embeddings_array, 'anger management')


#########

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(embeddings_array)
vis_dims.shape



# Use PCA for faster dimensionality reduction
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings_array)

# Add to df
df_posters['pca1'] = embeddings_2d_pca[:,0]
df_posters['pca2'] = embeddings_2d_pca[:,1]

# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Add to df
df_posters['tSNE1'] = embeddings_2d[:,0]
df_posters['tSNE2'] = embeddings_2d[:,1]

##################
# Dimensionality Reduction and Plot

# Randomly select embeddings
np.random.seed(442)  # Ensure reproducibility
random_indices = np.random.choice(embeddings_array.shape[0], size=100, replace=False)
selected_embeddings = embeddings_array[random_indices]
selected_titles = df_posters['*Session Title'].iloc[random_indices].values

#########
# Use PCA for faster dimensionality reduction
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(selected_embeddings)

# Edit titles to remove the initial "[number]" portion
edited_titles = [re.sub(r"^\[\d+\]", "", title).strip() for title in selected_titles]

# Function to wrap text based on a character limit
def wrap_text(text, limit):
    """
    Wrap text based on a given character limit.
    Splits the text into lines with lengths as close to the limit as possible
    without splitting words.
    """
    import textwrap
    return '<br>'.join(textwrap.wrap(text, width=limit))

# Apply text wrapping to the edited titles
wrapped_titles = [wrap_text(title, 30) for title in edited_titles]


# Create a plot with hover text using PCA results and modified titles
# This time, ensure that the titles are only visible as tooltips and not as static text on the plot
fig_pca_final = px.scatter(x=embeddings_2d_pca[:,0], y=embeddings_2d_pca[:,1],
                           hover_name=wrapped_titles,
                           title="2D Visualization of Random Session Embeddings with PCA")

# Update traces to customize hover information (only show wrapped titles, no x and y values)
fig_pca_final.update_traces(hovertemplate='%{hovertext}')

# Update layout
fig_pca_final.update_layout(hovermode="closest")
fig_pca_final.update_layout(xaxis_title="PCA Component 1", yaxis_title="PCA Component 2",
                            title=dict(text="2D Visualization of Random Session Embeddings with Titles on Hover using PCA"))

# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(selected_embeddings)

fig_pca_final = px.scatter(x=embeddings_2d[:,0], y=embeddings_2d[:,1],
                           hover_name=wrapped_titles,
                           title="2D Visualization of Random Session Embeddings with PCA")

# Update traces to customize hover information (only show wrapped titles, no x and y values)
fig_pca_final.update_traces(hovertemplate='%{hovertext}')

# Update layout
fig_pca_final.update_layout(hovermode="closest")
fig_pca_final.update_layout(xaxis_title="PCA Component 1", yaxis_title="PCA Component 2",
                            title=dict(text="2D Visualization of Random Session Embeddings with Titles on Hover using T-SNE"))

df_posters.to_csv('output/spsp_wEmbeddings.csv', index=False)

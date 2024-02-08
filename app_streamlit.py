# Imports
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import ast
import openai
import os

#############
# LOAD DATA #
#############

# Load the DataFrame
# Assume 'tSNE1' and 'tSNE2' are already in df
df = pd.read_csv('output/spsp_wEmbeddings.csv')

# Load embeddings
loaded_arrays = np.load('output/embeddings_array.npz')
embeddings_array = loaded_arrays['embeddings_array']

# openai.api_key = os.getenv('OPENAI_API_KEY') # local testing
openai.api_key = st.secrets['OPENAI_API_KEY']

from openai import OpenAI
client = OpenAI()

#############
# FUNCTIONS #
#############

# Function to get embeddings
def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# Function to search posters based on query_text
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

# Function to wrap text based on a character limit
def wrap_text(text, limit):
    """
    Wrap text based on a given character limit.
    Splits the text into lines with lengths as close to the limit as possible
    without splitting words.
    """
    import textwrap
    return '<br>'.join(textwrap.wrap(text, width=limit))

#############
# SIDEBAR   #
#############

# Adding a title to the sidebar
st.sidebar.markdown('''
                    # 🕵️ Poster Finder 
                    ''')

# Additional sidebar elements can be added below
st.sidebar.write("Find posters that might be similar to yours (or your interests).")
st.sidebar.divider()

# Option for users to choose input method
input_method = st.sidebar.radio("🔍 Search modality", ('Poster Title', 'Keywords'),
                                # label_visibility = "hidden"
                                )
# Make radio button label larger
st.markdown(
    """<style>
div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 18px;
    font-weight: 600;
}
    </style>
    """, unsafe_allow_html=True)

if input_method == 'Keywords':
    # Create a text input box in the sidebar
    query_text = st.sidebar.text_input("Enter keywords:", "")
    selected_title = None
else:
    # Create a select box with session titles
    selected_title = st.sidebar.selectbox(
        "Enter poster name (autocomplete):",
        options=df['*Session Title'].unique(),
        index=None,
        placeholder = 'Start typing here...',
        # index=0,  # Default to the first entry
        # format_func=lambda x: x[:50]  # Optionally format the display of titles if they are very long
    )
    query_text = None

st.sidebar.divider()
st.sidebar.markdown('''
                    Poster Finder v 0.1  
                    By [Daniel J Wilson](https://www.danieljwilson.com)
                    '''
        )
# Social media profile URLs
github_repo_url = "https://github.com/danieljwilson/"
twitter_profile_url = "https://twitter.com/_danieljwilson_"
linkedin_profile_url = "https://www.linkedin.com/in/danieljwilson/"

# Social media icon URLs
github_icon_url = "https://cdn4.iconfinder.com/data/icons/social-media-2231/512/71-github_social-512.png"
twitter_icon_url = "https://cdn1.iconfinder.com/data/icons/social-media-rounded-corners/512/Rounded_Twitter5_svg-1024.png"
linkedin_icon_url = "https://cdn1.iconfinder.com/data/icons/social-media-rounded-corners/512/Rounded_Linkedin2_svg-1024.png"

# HTML code with social links
html_code = f'''
    <a href="{github_repo_url}" target="_blank"><img src="{github_icon_url}" alt="GitHub" style="width:20px;height:20px;margin-right:5px;"></a>
    <a href="{twitter_profile_url}" target="_blank"><img src="{twitter_icon_url}" alt="Twitter" style="width:20px;height:20px;margin-right:5px;"></a>
    <a href="{linkedin_profile_url}" target="_blank"><img src="{linkedin_icon_url}" alt="Twitter" style="width:20px;height:20px;"></a>
    '''

# Display the GitHub icon with a link
st.sidebar.markdown(html_code, unsafe_allow_html=True)

########
# MAIN #
########
# Main logic to determine the source of plot_df based on user input
if query_text:
    query = True
    # Run search based on the user's query
    plot_df = search_posters(df, embeddings_array, query_text)
    
elif selected_title:
    query = False
    selected_row = df[df['*Session Title'] == selected_title]
    if not selected_row.empty:
        selected_index = selected_row.index[0]
        
        # Calculate cosine similarity
        array_list = ast.literal_eval(df['embedding_3small'][selected_index]) # Convert string to list
        selected_embedding = np.array(array_list).reshape(1, -1)
        similarities = cosine_similarity(embeddings_array, selected_embedding).flatten()
        df['Similarity'] = similarities # Store similarities in the original DataFrame
        
        # Get indices for 10 most similar, 5 most different, and 5 random entries
        similar_indices = np.argsort(similarities)[-11:]  # Last 10 plus the selected one itself
        different_indices = np.argsort(similarities)[:5]
        random_indices = np.random.choice(df.index.difference(similar_indices).difference(different_indices), 5, replace=False)
        
        combined_indices = np.unique(np.concatenate((similar_indices, different_indices, random_indices, [selected_index])))
        
        #  Subset DF
        plot_df = df.loc[combined_indices].copy()
        
        # Create names of groups
        plot_df['group'] = ''
        
        plot_df.loc[list(similar_indices), 'group'] = 'Similar'
        plot_df.loc[list(different_indices), 'group'] = 'Different'
        plot_df.loc[list(random_indices), 'group'] = 'Random'
        plot_df.loc[list([selected_index]), 'group'] = 'Original'
        # Order groups        
        plot_df['group'] = pd.Categorical(plot_df['group'], categories=['Original', 'Similar', 'Different', 'Random'], ordered=True)
else:
    st.write("Please select a poster name or enter a search term to begin.")
    plot_df = None  # Ensure plot_df is defined even if no input is given

if plot_df is not None:        
    # Define your color mapping for the 'continent' column categories
    color_map = {
        'Original': '#793FDF',
        'Similar': '#7091F5',
        'Random': '#FFB6D9',
        'Different': '#FFC436'
        }
    
    # Wrap title text
    plot_df.loc[:, 'wrapped_title'] = [wrap_text(title, 50) for title in plot_df['*Session Title']]
    
    if query == False:
        # Display session details below the plot
        selected_session = df.loc[selected_index] 

        # Define the content with the selected session title
        selected_session_title = selected_session['*Session Title']

        st.markdown(f"**{selected_session_title}**")
    elif query == True:
        st.markdown(f"**{query_text}**")
    
    #  Specify the desired order of 'group' values
    group_order = ['Original', 'Similar', 'Random', 'Different']
    
    # Generate plot        
    fig = px.scatter(plot_df, x='tSNE1', y='tSNE2',
                        #size='size',
                        color='group', # Column to define colors
                        color_discrete_map=color_map, # Apply color map
                        hover_data=['wrapped_title', 'Authors', '*Date', '*Time Start', '*Time End'],
                        category_orders={'group': group_order})  # Control legend order

    # Update hover template for all traces
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br><br>✍️ %{customdata[1]}<br>📅 %{customdata[2]}: %{customdata[3]} - %{customdata[4]}<extra></extra>")
    
    # Updating axis titles
    fig.update_layout(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title_text='Poster'
    )
    # Display plot
    st.plotly_chart(fig)
    
    # Display a DataFrame view with specified columns
    columns_to_display = ['group', 'Similarity', '*Session Title', 'Authors', '*Date', '*Time Start', '*Time End', 'Description']
    styled_df = plot_df[columns_to_display]
    # Sorting the DataFrame by the 'group' column
    styled_df = styled_df.sort_values(by='Similarity', ascending=False)
    
    # Function to apply row-wise coloring
    def color_rows(s):
        # Assuming 'group' is in the index or a column in the DataFrame
        if s.name == 'group':
            return [f'background-color: {color_map.get(group_val)}' if group_val in color_map else '' for group_val in plot_df['group']]

    # Apply the coloring function
    styled_df = styled_df.style.apply(color_rows)
    
    # Put df in expander
    with st.expander("View Table"):
        edited_df = st.data_editor(
            styled_df,
            hide_index=True,
            column_config={
                "group": "Group",
                "Similarity": st.column_config.NumberColumn(
                    "Similarity",
                    help="Calculated using cosine similarity on original 1536 element embedding vector"
                ),
                "*Session Title": "Title",
                "*Date": "Date",
                "*Time Start": "Start",
                "*Time End": "End"
            }
            )
    with st.expander("About Poster Finder"):
                '''
                **Poster Finder** uses OpenAI's recent [`text-embedding-3-small`](https://openai.com/blog/new-embedding-models-and-api-updates) embedding model
                to create vector embeddings with 1536 dimensions of each poster's description text.
                
                Comparisons are made by comparing the selected poster's description embedding to all other
                posters' description embeddings using [cosine similarity](https://www.pinecone.io/learn/vector-similarity/#Cosine-Similarity)
                which measures the angle between vectors in multidimensional space.
                
                For plotting I use [t-distributed Stochastic Neighbor Embedding](https://www.datacamp.com/tutorial/introduction-t-sne) (t-SNE)
                for dimensions reduction and then take the first two components
                to determine the x and y positions.
                
                ---
                The goal is to provide a (hopefully) handy glance at posters that may be in your wheelhouse.
                
                All code is published on a [Github repo](https://github.com/danieljwilson/LLM-Poster-Recommender), and should be easy to repurpose for any conference.
                
                🐛 This is a very beta version so expect glitches.
                
                ❓ Please feel free to [email me](mailto:daniel.j.wilson@gmail.com) with any issues, feedback, or questions.
                '''

import streamlit.components.v1 as components


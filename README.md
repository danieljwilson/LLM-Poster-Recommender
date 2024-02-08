# 🕵️ Poster Finder

## Overview
This project comprises a Jupyter notebook for generating embeddings from poster descriptions (`poster_embeddings.ipynb`) and a Streamlit application (`app_streamlit.py`) for visualizing these embeddings. It aims to provide an interactive platform to explore semantic similarities between posters based on their descriptions using OpenAI's embedding models.

## Features
- **Embedding Generation**: Calculate embeddings for poster descriptions using OpenAI's API.
- **Dimensionality Reduction**: Apply PCA and t-SNE for embedding visualization.
- **Streamlit Web App**: Interactive UI to search and visualize posters by similarity.

## Prerequisites
- Python 3.6 or higher
- Streamlit
- OpenAI
- Pandas, NumPy, Scikit-learn, Plotly

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/danieljwilson/LLM-Poster-Recommender.git
   cd LLM-Poster-Recommender
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # For Windows
   venv\Scripts\activate

   # For Unix or MacOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set the OpenAI API key:
   - **Locally**: Set an environment variable `OPENAI_API_KEY` with your OpenAI API key.
     ```bash
     # For Windows
     setx OPENAI_API_KEY "your_openai_api_key_here"
     # For Unix or MacOS
     export OPENAI_API_KEY="your_openai_api_key_here"
     ```
   - **Streamlit Sharing/Deployment**: Add your OpenAI API key as a secret. In Streamlit Sharing, navigate to your app settings, find the Secrets section, and add your OpenAI API key like so:
     ```yaml
     OPENAI_API_KEY: "your_openai_api_key_here"
     ```

## Running the App Locally

1. Generate embeddings:
   
   Use the `poster_embeddings.ipynb` notebooks to processes the poster descriptions, generates embeddings, and saves them along with dimensional reductions for visualization.

2. Start the Streamlit app:
   ```bash
   streamlit run app_streamlit.py
   ```
   The Streamlit app provides an interface to visualize the posters' embeddings and explore their similarities.

## Deployment

To deploy the Streamlit app, you can use services like Streamlit Sharing, Heroku, or any platform supporting Python apps. Ensure you add your `OPENAI_API_KEY` as a secret or environment variable according to the platform's guidelines.

### Streamlit Sharing
1. Push your code to a GitHub repository.
2. Sign up for Streamlit Sharing and connect your GitHub account.
3. Select your repository and branch, then click on deploy.
4. Add your `OPENAI_API_KEY` in the app's settings under Secrets.

### Heroku
1. Create a `Procfile` and add:
   ```
   web: sh setup.sh && streamlit run app_streamlit.py
   ```
2. Create a `setup.sh` script to set environment variables (including `OPENAI_API_KEY`) and start the app.
3. Follow Heroku's documentation to deploy the app.

## Contributing
Contributions are welcome. Please fork the repository and submit pull requests with your changes.

## License
[MIT](https://github.com/danieljwilson/LLM-Poster-Recommender/blob/main/LICENSE).

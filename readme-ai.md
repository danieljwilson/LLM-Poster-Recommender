<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
</p>
<p align="center">
    <h1 align="center">LLM-POSTER-RECOMMENDER</h1>
</p>
<p align="center">
    <em>Poster Magic: Your Personalized Recommendation Assistant</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/danieljwilson/LLM-Poster-Recommender?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/danieljwilson/LLM-Poster-Recommender?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/danieljwilson/LLM-Poster-Recommender?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/danieljwilson/LLM-Poster-Recommender?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black" alt="tqdm">
	<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat&logo=Pydantic&logoColor=white" alt="Pydantic">
	<img src="https://img.shields.io/badge/Jinja-B41717.svg?style=flat&logo=Jinja&logoColor=white" alt="Jinja">
	<br>
	<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
	<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat&logo=OpenAI&logoColor=white" alt="OpenAI">
	<img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style=flat&logo=Plotly&logoColor=white" alt="Plotly">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Features](#-features)
> - [ Repository Structure](#-repository-structure)
> - [ Modules](#-modules)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [Running LLM-Poster-Recommender](#-running-LLM-Poster-Recommender)
>   - [ Tests](#-tests)
> - [ Project Roadmap](#-project-roadmap)
> - [ Contributing](#-contributing)
> - [ License](#-license)
> - [ Acknowledgments](#-acknowledgments)

---

##  Overview

The LLM-Poster-Recommender project is a web application that provides users with personalized poster recommendations. By utilizing OpenAI's text embedding model and cosine similarity, it allows users to search and recommend similar posters based on query text or selected poster titles. The application displays the results in a scatter plot and a table, making it easy for users to explore and find relevant posters. The project also includes a module for loading and processing tabular data, obtaining embeddings using the OpenAI API, and performing dimensionality reduction using PCA and t-SNE. Overall, the LLM-Poster-Recommender project simplifies the search and discovery process for users, providing them with tailored and relevant poster recommendations.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | The architecture of the LLM-Poster-Recommender project is not explicitly described in the provided information. It utilizes Streamlit for the web interface and OpenAI's text embedding model for recommendation. |
| 🔩 | **Code Quality**  | The code quality and style of the LLM-Poster-Recommender project are not explicitly mentioned in the provided information. |
| 📄 | **Documentation** | The extent and quality of documentation in the LLM-Poster-Recommender project are not explicitly mentioned in the provided information. |
| 🔌 | **Integrations**  | The key integrations and external dependencies of the LLM-Poster-Recommender project include Plotly Express, GitPython, Altair, scikit-learn, pandas, Streamlit, and OpenAI's text embedding model. |
| 🧩 | **Modularity**    | The modularity and reusability of the codebase in the LLM-Poster-Recommender project are not explicitly mentioned in the provided information. |
| 🧪 | **Testing**       | The testing frameworks and tools used in the LLM-Poster-Recommender project are not explicitly mentioned in the provided information. |
| ⚡️  | **Performance**   | The efficiency, speed, and resource usage of the LLM-Poster-Recommender project are not explicitly mentioned in the provided information. |
| 🛡️ | **Security**      | The measures used for data protection and access control in the LLM-Poster-Recommender project are not explicitly mentioned in the provided information. |
| 📦 | **Dependencies**  | The key external libraries and dependencies of the LLM-Poster-Recommender project include Plotly Express, GitPython, Altair, scikit-learn, pandas, Streamlit, and OpenAI's text embedding model. |


---

##  Repository Structure

```sh
└── LLM-Poster-Recommender/
    ├── app_streamlit.py
    ├── output
    │   ├── embeddings_array.npz
    │   └── spsp_wEmbeddings.csv
    ├── poster_embeddings.ipynb
    └── requirements.txt
```

---

##  Modules

<details closed><summary>.</summary>

| File                                                                                                                   | Summary                                                                                                                                                                                                                                                                                                                                                             |
| ---                                                                                                                    | ---                                                                                                                                                                                                                                                                                                                                                                 |
| [requirements.txt](https://github.com/danieljwilson/LLM-Poster-Recommender/blob/master/requirements.txt)               | This code snippet, located in the app_streamlit.py file, is part of the larger LLM-Poster-Recommender repository. It contributes to the architecture of the repository by providing the main application logic for the Streamlit-based web interface. Its critical features include displaying poster recommendations and utilizing pre-computed poster embeddings. |
| [app_streamlit.py](https://github.com/danieljwilson/LLM-Poster-Recommender/blob/master/app_streamlit.py)               | This code snippet is part of the LLM-Poster-Recommender repository. It is responsible for searching and recommending similar posters based on query text or selected poster titles. The code utilizes OpenAI's text embedding model and cosine similarity for comparison. The results are displayed in a scatter plot and a table.                                  |
| [poster_embeddings.ipynb](https://github.com/danieljwilson/LLM-Poster-Recommender/blob/master/poster_embeddings.ipynb) | The code snippet in `poster_embeddings.ipynb` is responsible for loading and processing tabular data, filtering and editing the data, obtaining embeddings using the OpenAI API, and performing dimensionality reduction using PCA and t-SNE. The resulting embeddings and reduced dimensions are saved in the output files.                                        |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **JupyterNotebook**: `version x.y.z`

###  Installation

1. Clone the LLM-Poster-Recommender repository:

```sh
git clone https://github.com/danieljwilson/LLM-Poster-Recommender
```

2. Change to the project directory:

```sh
cd LLM-Poster-Recommender
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

###  Running `LLM-Poster-Recommender`

Use the following command to run LLM-Poster-Recommender:

```sh
jupyter nbconvert --execute notebook.ipynb
```

###  Tests

Use the following command to run tests:

```sh
pytest notebook_test.py
```

---

##  Project Roadmap

- [X] `► INSERT-TASK-1`
- [ ] `► INSERT-TASK-2`
- [ ] `► ...`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Submit Pull Requests](https://github.com/danieljwilson/LLM-Poster-Recommender/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/danieljwilson/LLM-Poster-Recommender/discussions)**: Share your insights, provide feedback, or ask questions.
- **[Report Issues](https://github.com/danieljwilson/LLM-Poster-Recommender/issues)**: Submit bugs found or log feature requests for the `LLM-Poster-Recommender` project.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/danieljwilson/LLM-Poster-Recommender
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-quick-links)

---

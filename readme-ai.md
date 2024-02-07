<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
</p>
<p align="center">
    <h1 align="center">LLM-POSTER-RECOMMENDER</h1>
</p>
<p align="center">
    <em>Empowering discovery through personalized poster recommendations.</em>
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
	<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat&logo=Pydantic&logoColor=white" alt="Pydantic">
	<img src="https://img.shields.io/badge/Jinja-B41717.svg?style=flat&logo=Jinja&logoColor=white" alt="Jinja">
	<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
	<br>
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

LLM-Poster-Recommender is an innovative project that aims to enhance the poster selection process by providing users with personalized recommendations. With the help of deep learning techniques, this codebase generates poster embeddings, which represent the unique characteristics of each poster. Users can search for posters based on keywords or selected titles, and the code fetches and displays similar posters. Additionally, it leverages t-SNE dimension reduction to create a scatter plot, visually demonstrating the similarity between the posters. By integrating machine learning algorithms, LLM-Poster-Recommender offers a valuable solution for individuals and businesses seeking efficient poster exploration and discovery.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | The project's architecture consists of several code files that contribute to the overall functionality. It includes code for generating poster embeddings and code for searching and displaying similar posters based on input keywords or selected titles. The architecture utilizes an API for text embeddings and applies dimensionality reduction techniques using PCA and t-SNE. |
| 🔩 | **Code Quality**  | The code quality and style are not mentioned in the provided codebase details. |
| 📄 | **Documentation** | The extent and quality of documentation are not mentioned in the provided codebase details. |
| 🔌 | **Integrations**  | The key integration in the project is the use of an API for obtaining text embeddings. It also has dependencies on various external libraries, including HTTPX, Pillow, Altair, Plotly, Streamlit, Pandas, Scikit-learn, and more. |
| 🧩 | **Modularity**    | The codebase appears to be modular as it consists of separate code files for different functionalities, such as generating embeddings, processing data, and searching/displaying similar posters. This modularity promotes code reusability. |
| 🧪 | **Testing**       | The testing frameworks and tools used are not mentioned in the provided codebase details. |
| ⚡️  | **Performance**   | The efficiency, speed, and resource usage of the project are not mentioned in the provided codebase details. |
| 🛡️ | **Security**      | The measures used for data protection and access control are not mentioned in the provided codebase details. |
| 📦 | **Dependencies**  | The key external libraries and dependencies include HTTPX, Pillow, Altair, Plotly, Streamlit, Pandas, Scikit-learn, and more. |


---

##  Repository Structure

```sh
└── LLM-Poster-Recommender/
    ├── app_streamlit.py
    ├── output
    │   ├── embeddings_array.npz
    │   └── spsp_wEmbeddings.csv
    ├── poster_embeddings.py
    └── requirements.txt
```

---

##  Modules

<details closed><summary>.</summary>

| File                                                                                                             | Summary                                                                                                                                                                                                                                                                                                                                                                           |
| ---                                                                                                              | ---                                                                                                                                                                                                                                                                                                                                                                               |
| [requirements.txt](https://github.com/danieljwilson/LLM-Poster-Recommender/blob/master/requirements.txt)         | This code snippet, located within the `LLM-Poster-Recommender` repository, contributes to the architecture by providing poster embeddings. The code generates embeddings for posters and stores them in output files.                                                                                                                                                             |
| [app_streamlit.py](https://github.com/danieljwilson/LLM-Poster-Recommender/blob/master/app_streamlit.py)         | This code snippet is part of the LLM Poster Recommender repository. It defines functions to search for and display similar posters based on input keywords or selected titles. The code also creates a scatter plot to visualize the similarity between posters using t-SNE dimension reduction.                                                                                  |
| [poster_embeddings.py](https://github.com/danieljwilson/LLM-Poster-Recommender/blob/master/poster_embeddings.py) | The code snippet in the file poster_embeddings.py is responsible for processing data, obtaining text embeddings, performing dimensionality reduction using PCA and t-SNE, and saving the results to a CSV file. It filters and processes data from an Excel file, accesses an API for text embeddings, applies dimensionality reduction techniques, and saves the processed data. |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version x.y.z`

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
python main.py
```

###  Tests

Use the following command to run tests:

```sh
pytest
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

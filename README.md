# 🌙 Sleep Health Chatbot

## 📝 Exploring Personalized Health Support through Data-Driven, Theory-Guided LLMs: A Case Study in Sleep Health (CHI' 25)

![Chatbot Framework](image/chatbot_framework.png)

[![Arxiv](https://img.shields.io/badge/Arxiv-1B1B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2502.13920) | 
[![ACM DL](https://img.shields.io/badge/ACM_DL-0085CA?style=for-the-badge&logo=acm&logoColor=white)](https://dl.acm.org/doi/10.1145/3706598.3713852)

**Contact**: 📧 Xingbo Wang (wangxbzb@gmail.com)

This code base demonstrates a LLM-powered sleep health chatbot augmented by a multiagent framework to provide personalized, data-driven, and theory-guided sleep health support. The chatbot integrates wearable device data, contextual information, and established behavior change theories to deliver adaptive recommendations and motivational support.

## 📁 Project Structure

- `ThSleepHealthBot.py` - The main file for the chatbot interface and user interaction
- `agent_coordinator.py` - Coordinates multiple specialized agents (e.g., Recommendation, Health Data Analysis)
- `activityRec.py` - Contextual multi-armed bandit model for activity recommendation
  - `generate_activity` - Generate synthetic data for activity recommendations research demonstration
- `utils.py` - Utility functions for data retrieval and processing
- `globalVariable.py` - Global variables for the project
- `requirements.txt` - The dependencies for the project
- `README.md` - The readme file for the project
- `.streamlit` - The streamlit secret configuration file: `.streamlit/secrets.toml`

## 🚀 Getting Started

### 📋 Prerequisites
- Python 3.10.13

### 💻 Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### ⚙️ Environment Setup

Create a `.streamlit/secrets.toml` file in the `.streamlit` directory with the following configuration:
- We have provided an example of recommendation model in the parent directory: `user_model.pkl`

```toml
[oura]
oura_token = "input_your_oura_token"

[weatherapi]
weatherapi_key = "input_your_weatherapi_key"

[user]
pid = 1
user_name = "johndoe"
user_model_path = "user_model.pkl"
```

Required environment variables:
- `OPENAI_API_KEY` - The API key for the OpenAI API
- `oura_token` - The token for the Oura API
- `weatherapi_key` - The API key for the [WeatherAPI](https://www.weatherapi.com/)
- User information:
  - `pid` - Participant ID
  - `user_name` - User name
  - `user_model_path` - Path to the user model file (An example user model is provided)

### 🏃‍♂️ Running the Chatbot

Run the chatbot using:
```bash
streamlit run ThSleepHealthBot.py
```


## 📚 Citation

```bibtex
@inproceedings{sleepllm_chi25,
  author = {Wang, Xingbo and Griffith, Janessa and Adler, Daniel A. and Castillo, Joey and Choudhury, Tanzeem and Wang, Fei},
  title = {Exploring Personalized Health Support through Data-Driven, Theory-Guided LLMs: A Case Study in Sleep Health},
  year = {2025},
  isbn = {9798400713941},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3706598.3713852},
  doi = {10.1145/3706598.3713852},
  articleno = {507},
  numpages = {15},
  series = {CHI '25}
}
```




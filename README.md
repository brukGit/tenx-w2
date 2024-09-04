
# Telco dataset

FNSPID (Financial News and Stock Price Integration Dataset), is a comprehensive financial dataset designed to enhance stock market predictions by combining quantitative and qualitative data.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Exploratory Data Analysis (EDA) Overview](#exploratory-data-analysis-eda-overview)
- [Data Cleaning](#data-cleaning)
- [Dashboard](#dashboard)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up this project on your local machine, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/brukGit/tenx-w2.git
   cd notebooks

2. **Checkout branch task-1**:
  ```bash
   git checkout task-2

3. **Create a virtual environment (optional but recommended)**:
    ```bash
    python3 -m venv venv
    source venv\Scripts\activate  # On Linux, use `venv/bin/activate`

4. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt


## Project Structure
    ```bash
        ├── data/                  # Directory containing raw datasets
        ├── notebooks/             # Jupyter notebooks for EDA and analysis
        ├── scripts/               # Python scripts for data processing and visualization
        ├── tests/                 # Unit tests for the project
        ├── app/                   # Interactive app built with streamlit
        ├── .github/workflows/     # GitHub Actions for CI/CD
        ├── .vscode/               # Visual Studio Code settings and configurations
        ├── requirements.txt       # Python dependencies
        ├── README.md              # Project documentation (this file)
        └── LICENSE                # License for the project


## Usage
### Running the Notebooks
To perform the EDA, navigate to the notebooks/ directory and open the provided Jupyter notebook. The notebook focuses on different aspects of the analysis, including descriptive statistics for stock indicators, time series analysis, and financial metrics analysis using TA-Lib and yfiance.
    ```bash
    jupyter notebook notebooks/anayze_stocks.ipynb
   

### Running the Notebooks
You can use the script 'analyze_stocks.py' inside scripts directory to run all scripts located in 'src/' directory. Just change directory to scripts and executed the script inside. 
    ```bash
    cd scripts
    python run analyze_stocks.py

### Running Tests
If you want to run unit tests to ensure that the functions work as expected (although, sorry, currently no test code is provided.):
    
```bash
    python -m unittest discover -s tests

Use the sample csv data provided in sample_data directory for testing purposes.

### Raw datasets
Add your datasets inside data directory.

## Exploratory Data Analysis (EDA) Overview
The EDA conducted in this project covers several key areas:

○	Indicator calculations for RSI, MA, MCDA, MCDA signal and MCDA histogram and visualizations from their summary statistics
○	Indicators versus stock prices analysis with visualizations using heatmap and time series plots
○	Correlation analysis of returns among companies (tickers) using time-series and correlation heatmaps
○	Financial metrics analysis of volatility, Sharpe Ratio and Beta for all companies.

## Data Cleaning
Based on the initial analysis, the dataset was cleaned by handling missing values, removing duplicates, and ensuring correct data types.

## Dashboard
Visit the dashboard built with streamlit on https:// ...
or change your directory to app and run the following command.
```bash
    streamlit run main.py
## Contributing
Contributions to this project are welcome! If you have suggestions or improvements, feel free to open a pull request or issue on GitHub.

## License
This project is licensed under the MIT License - see the LICENSE file for details.



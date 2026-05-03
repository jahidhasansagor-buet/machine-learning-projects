# Mall Customer Segmentation

A machine learning web application that performs customer segmentation using K-Means clustering on mall customer data. Built with Streamlit for interactive data exploration and visualization.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Live Demo

**Try the app here:** [https://jahid-mall-customer-segmentation.streamlit.app](https://jahid-machine-learning-projects-mall-customer-segmentation.streamlit.app)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Learning Outcomes](#learning-outcomes)
- [Author](#author)

## Overview

This project implements customer segmentation using K-Means clustering to identify distinct groups of mall customers based on their annual income and spending score. The interactive web application helps businesses understand their customer base and develop targeted marketing strategies.

### Key Objectives:
- Segment customers into meaningful groups
- Identify high-value customers
- Understand spending patterns
- Support data-driven marketing decisions

## Features

### Core Features
- **Dataset Overview**: Comprehensive dataset statistics and preview
- **Elbow Method Analysis**: Visual tool to determine optimal number of clusters
- **K-Means Clustering**: Interactive clustering with adjustable K value (2-10)
- **Cluster Visualization**: Scatter plots showing customer segments
- **Live Deployment**: Accessible via public Streamlit Cloud URL

### Advanced Features
- **Automatic K Suggestion**: Uses KneeLocator algorithm for optimal cluster detection
- **Cluster Profiles**: Detailed statistical analysis of each customer segment
- **Gender Distribution**: Gender breakdown analysis per cluster
- **Smart Insights**: Automatic categorization of clusters (High Value, Budget Conscious, etc.)
- **Export Functionality**: Download clustered data and analysis results as CSV

## Dataset

**File**: `Mall_Customers.csv`

### Columns:
| Column | Description |
|--------|-------------|
| CustomerID | Unique ID assigned to each customer |
| Gender | Gender of the customer (Male/Female) |
| Age | Age of the customer |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Score assigned based on spending behavior and patterns |

### Dataset Statistics:
- **Total Customers**: 200
- **Features**: 5
- **Target Variables for Clustering**: Annual Income, Spending Score

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/jahidhasansagor-buet/machine-learning-projects.git
cd machine-learning-projects/Mall_Customer_Segmentation
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
streamlit --version
```

## Usage

### Run Locally
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Dataset Overview**: View basic statistics and data preview
2. **Elbow Method**: Analyze the elbow curve to identify optimal K
3. **Adjust K Value**: Use the sidebar slider to select number of clusters
4. **View Results**: Explore cluster visualizations and profiles
5. **Download Data**: Export clustered dataset and analysis results

### Key Controls
- **K Value Slider** (Sidebar): Adjust the number of clusters (2-10)
- **Expandable Sections**: Click to view detailed dataset information
- **Download Buttons**: Export results as CSV files

## Deployment

This app is deployed on **Streamlit Cloud** and accessible at:
**[https://jahid-mall-customer-segmentation.streamlit.app](https://jahid-machine-learning-projects-mall-customer-segmentation.streamlit.app)**

### Deploy Your Own Version

1. **Fork this Repository**
   - Click "Fork" at the top right of this repo

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click **"New app"**
   - Fill in the deployment form:
     - **Repository**: `jahidhasansagor-buet/machine-learning-projects`
     - **Branch**: `main`
     - **Main file path**: `Mall_Customer_Segmentation/app.py`
   - Click **"Deploy"**

3. **Wait for Deployment**
   - Deployment typically takes 2-3 minutes
   - You'll receive a public URL

### Environment Requirements
All dependencies are specified in `requirements.txt` and will be automatically installed during deployment.

## Project Structure

```
Mall_Customer_Segmentation/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Mall_Customers.csv          # Dataset
├── README.md                   # Project documentation
├── QUICK_START.md             # Quick deployment guide
│
└── .gitignore                  # Git ignore file
```

## Technologies Used

### Core Technologies
- **Python 3.8+**: Programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning (K-Means clustering)

### Visualization
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization

### Additional Libraries
- **kneed**: Automatic elbow detection using KneeLocator algorithm


## License

This project is licensed under the MIT License.

## Author

**Jahid Hasan Sagor**
- GitHub: [@jahidhasansagor-buet](https://github.com/jahidhasansagor-buet)
- Project: [Machine Learning Projects](https://github.com/jahidhasansagor-buet/machine-learning-projects)

## Acknowledgments

- Dataset provided as part of machine learning coursework


# Travel Insurance Prediction Analysis

## **Project Overview**
This repository contains an analysis of a Travel Insurance Prediction dataset, aiming to identify key factors influencing whether a customer purchases travel insurance. The project involves exploratory data analysis (EDA), statistical inference, and machine learning models, with a focus on evaluating different classification models to determine the most effective one for predicting insurance purchase likelihood.

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Analysis Overview](#analysis-overview)
7. [Key Findings](#key-findings)
8. [Improvements and Future Work](#improvements-and-future-work)
9. [Stakeholders and Goals](#stakeholders-and-goals)
10. [Contributors](#contributors)

## Introduction
The purpose of this analysis is to explore the factors influencing travel insurance purchase decisions. By leveraging machine learning models and statistical techniques, this study aims to provide actionable insights for insurance companies to optimize their marketing strategies and improve customer targeting.

## Setup

### Prerequisites
- Python 3.x
- Poetry for dependency management
- Jupyter Notebook (optional, for viewing the analysis)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Vixamon/travel_insurance_predictor/
   cd travel_insurance_predictor
   ```

2. Set up a virtual environment and install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

4. (Optional) If using Jupyter Notebook to view or modify the analysis:  
   ```bash
   poetry add notebook
   ```

## Project Structure

- `pyproject.toml`: Poetry configuration file listing dependencies.
- `poetry.lock`: Lock file with exact package versions.
- `TravelInsurancePrediction.csv`: Dataset for the analysis.
- `travel_insurance_model_analysis.ipynb`: Jupyter notebook with the analysis and insights derived from the dataset.
- `utilities.py`: Contains helper functions for data preprocessing and visualization.

## Usage

### Running the Jupyter Notebook
To interact with the data analysis or run your own queries, use the Jupyter notebook:
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook travel_insurance_model_analysis.ipynb
   ```
2. Follow the cells to explore the analysis or modify them to perform your own exploration.

## Dataset

### Travel Insurance Dataset

- **Description:** The dataset includes customer information, travel history, and insurance purchase behavior.
- **Columns:**
  - `Age`: Age of the customer.
  - `Employment Type`: The sector in which the customer is employed (Government Sector or Private Sector/Self Employed)
  - `GraduateOrNot`: Whether the customer is a college graduate or not.
  - `AnnualIncome`: The yearly income of the customer in Indian Rupees (rounded to nearest 50 thousand Rupees)
  - `FamilyMembers`: Number of members in customer's family
  - `ChronicDiseases`: Whether the customer suffers from any major disease or conditions like diabetes/high BP or asthama, etc.
  - `FrequentFlyer`: Derived data based on customer's history of booking air tickets on atleast 4 different instances in the years 2017-2019.
  - `EverTravelledAbroad`: Has the customer ever travelled to a foreign country (not necessarily using the company's services)
  - `TravelInsurance`: Did the customer buy travel insurance package during introductory offering held in the year 2019 (target variable).

## Analysis Overview

The analysis includes:

- **Exploratory Data Analysis (EDA):** Understanding feature distributions, missing values, and correlations.
- **Feature Engineering:** Creating new features and handling categorical variables.
- **Baseline Models:** Evaluating dummy classifiers and basic machine learning models.
- **Machine Learning Models:**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Trees
  - Random Forest Classifier
- **Model Evaluation:** Using confusion matrices, precision-recall, and other classification metrics.
- **Model Selection:** Comparing models and selecting the most effective one for prediction.

## Key Findings

- **Influential Variables:**
  - Higher income individuals tend to buy travel insurance more frequently.
  - Customers who have previously traveled abroad are more likely to opt for insurance.

- **Model Performance:**
  - Random Forest and SVM performed the best among tested models.
  - The best model (Random Forest) was selected based on PR-AUC score.

- **Insights:**
  - Travel history, income levels, age and family size are key predictors of insurance purchase behavior.
  - Insurance companies can target frequent travelers and high-income individuals with personalized offers.

## Improvements and Future Work

1. **Investigate feature importance in SVM & Logistic Regression:**
   - Understanding which features influence predictions in these models can provide deeper insights into customer behavior.

2. **Consider ensemble methods (combining Logistic Regression & Random Forest):**
   - Blending these models may enhance prediction accuracy by leveraging their strengths.

3. **Try more models after feature engineering:**
   - Testing additional algorithms, such as gradient boosting, can further improve predictive performance.
   
4. **Check class imbalance handling (resampling, SMOTE):**
   - Ensuring balanced classes can enhance model reliability and prevent bias toward majority classes.


## Stakeholders and Goals

### Stakeholders

- **Insurance Companies:** To maximize revenue by targeting real potential buyers.
- **Marketing Teams:** To optimize ad spending by reducing wasted targeting
- **Business Analysts:** To improve risk assessment for travel insurance policies.

### Goals

- Identify key predictors of travel insurance purchases.
- Develop a machine learning model that provides actionable insights.
- Suggest data-driven marketing strategies for insurance companies.

## Contributors
- [Erikas Leonenka](https://github.com/Vixamon)

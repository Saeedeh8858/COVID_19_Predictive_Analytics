# COVID-19 ML Predictive Analytics Project

## Overview

This project applies machine learning techniques to COVID-19 patient data to answer key healthcare questions, such as predicting reinfection risk, ICU admission, hospitalization, recovery speed, ventilator need, vaccine type, and age group classification. The goal is to support healthcare stakeholders in resource planning and risk stratification.

## Dataset

- Source: [Kaggle COVID-19 Reinfection and Health Dataset](https://www.kaggle.com/datasets/khushikyad001/covid-19-reinfection-and-health-dataset)
- Features include: Demographics, vaccination status, symptoms, preexisting conditions, COVID strain, hospitalization, ICU admission, ventilator support, recovery status, and more.

## Key Questions Addressed

1. Predict the likelihood of COVID-19 reinfection.
2. Identify individuals most at risk of reinfection.
3. Predict ICU admission based on health indicators.
4. Evaluate hospitalization prediction models.
5. Predict fast recovery (<14 days).
6. Predict need for ventilator support.
7. Analyze vaccination factors and ICU risk.
8. Feature importance for ICU admission.
9. Predict vaccine type administered.
10. Predict age group (Youth, Adult, Elderly).

## ML Pipeline

- **Data Cleaning:** Handle missing values, outliers, duplicates, and encode categorical features.
- **EDA:** Visualize distributions and correlations.
- **Feature Engineering:** Create new features and interaction terms.
- **Modeling:** Train and evaluate multiple ML models (XGBoost, Random Forest, etc.).
- **Balancing:** Use SMOTE for class imbalance.
- **Evaluation:** Use accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

## Technologies Used

- Python (pandas, numpy, scikit-learn, imbalanced-learn, xgboost, matplotlib, seaborn)
- Jupyter Notebook / .py scripts
- Streamlit (for interactive dashboard)

## How to Run

1. Clone the repository.
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```
    python covid19_ml.py
    ```
4. (Optional) Launch the dashboard:
    ```
    streamlit run covid19_ml.py
    ```

## Results

- High accuracy in predicting reinfection, ICU admission, and hospitalization.
- Feature importance analysis guides public health interventions.
- Dashboard enables scenario analysis for healthcare decision-makers.

## Stakeholder Impact

- **Healthcare Providers:** Early identification of high-risk patients.
- **Hospitals/Clinics:** Resource allocation and surge planning.
- **Policy Makers:** Data-driven strategies for vaccination and resource distribution.
- **Public Health Officials:** Targeted interventions for vulnerable populations.

## License

This project is for educational and research purposes.

## Contact

For questions or collaboration, please contact [Saeedeh] via [LinkedIn](https://www.linkedin.com/in/saeedehalamkar) or [GitHub](https://github.com/Saeedeh8858/)



```markdown
# Housing Affordability – Model Dashboard

# Housing Affordability Dashboard

This project develops an interactive data analytics and machine learning solution to analyze housing affordability trends.

##  Project Overview
- Built an end-to-end analytics pipeline combining housing, economic, and demographic data  
- Developed predictive models to estimate affordability patterns  
- Deployed an interactive dashboard for real-time insights  

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- Streamlit
- Machine Learning: Random Forest, XGBoost
- Data Visualization

## Key Features
- Interactive dashboard for exploring housing affordability trends  
- Model-based predictions using machine learning  
- Data cleaning, transformation, and feature engineering  
- Visualization of key economic and housing indicators  

##  Business Value
- Supports data-driven decision-making in housing and policy analysis  
- Identifies affordability trends and potential risk areas  
- Helps users explore “what-if” scenarios interactively  

##  Deployment
- Streamlit dashboard for real-time interaction  

##  GitHub
https://github.com/AliReza0015-ux/housing-affordability-bi

---
Key Skills Demonstrated:
- Python data analysis
- Machine learning model comparison
- Streamlit dashboard development
- Classification metrics and feature importance
- Communicating insights through an interactive dashboard
## Project Structure
```

.
├── models/                                   # Saved trained models (optional)
│   ├── random\_forest.pkl
│   └── xgboost.pkl
├── outputs/                                  # Artifacts exported from Phase-2 Jupyter notebook
│   ├── features\_for\_modeling.csv
│   ├── random\_forest\_confusion\_matrix.png
│   ├── random\_forest\_feature\_importance\_top5.csv
│   ├── random\_forest\_feature\_importance\_top5.png
│   ├── random\_forest\_metrics.json
│   ├── xgboost\_confusion\_matrix.png
│   ├── xgboost\_feature\_importance\_top5.csv
│   ├── xgboost\_feature\_importance\_top5.png
│   ├── xgboost\_metrics.json
├── app.py                                    # Main Streamlit UI
├── model.py                                  # Functions to load metrics & artifacts
├── utils.py                                  # UI helpers & reusable functions
├── requirements.txt
└── README.md

````

---

#
##  How it Works

* The **Phase-2 Jupyter notebook** trains **Random Forest** and **XGBoost** models.
* Artifacts are saved to `outputs/`:

  * `*_metrics.json` – accuracy, F1 score, and classification report
  * `*_confusion_matrix.png` – confusion matrix plot
  * `*_feature_importance_top5.csv/png` – top 5 most important features
* The Streamlit app dynamically loads these artifacts and displays:

  * Metrics in **summary cards**
  * Confusion matrix plots
  * Feature importance chart & table

---

## Example Use Cases

* **Compare ML models** for classification tasks
* **Communicate results** to stakeholders with a simple dashboard
* **Demonstrate workflow** from data science notebook → deployable app

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

##  Author

Ali Reza Mohseni
URl=https://github.com/AliReza0015-ux/housing-affordability-bi


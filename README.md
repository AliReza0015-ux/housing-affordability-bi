

```markdown
# Housing Affordability – Model Dashboard

This project is part of **Phase-2** of the Housing Affordability Analysis.  
It provides an **interactive dashboard** built with [Streamlit](https://streamlit.io/) to explore the performance of **Random Forest** and **XGBoost** models trained to classify housing affordability levels.

---

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


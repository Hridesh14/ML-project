```markdown
🎓 Student Maths Score Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-green)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

📌 Project Overview
This project is an end-to-end **Machine Learning Web Application** that predicts a student's **Math Score** based on various demographic and academic factors. The goal is to understand how variables like parental education, test preparation, and lunch type influence academic performance.

The application is built using **Python, Flask, and Scikit-Learn**, featuring a complete pipeline from data ingestion to model deployment.

---

🛠️ Tech Stack
* **Frontend**: HTML5, CSS3, Bootstrap 5 (Responsive Design)
* **Backend**: Flask (Python Web Framework)
* **Machine Learning**: Scikit-Learn, Pandas, NumPy
* **Ensemble Models**: RandomForest, XGBoost, CatBoost, AdaBoost, GradientBoosting
* **Deployment**: Ready for AWS Elastic Beanstalk / Azure Web Apps

---

📂 Project Structure
```text
├── artifacts/              # Stores generated files (model.pkl, preprocessor.pkl)
├── notebook/               # Jupyter notebooks for EDA and experimentation
├── src/                    # Source code for the ML project
│   ├── components/         # Modules for Data Ingestion, Transformation, & Model Training
│   ├── pipeline/           # Prediction and Training pipelines
│   ├── utils.py            # Utility functions (save/load objects, evaluate models)
│   ├── logger.py           # Custom logging setup
│   └── exception.py        # Custom exception handling
├── templates/              # HTML templates for Flask
│   ├── index.html          # Landing page
│   └── home.html           # Prediction interface
├── app.py                  # Main Flask application entry point
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup file
└── README.md               # Project documentation

```

---

📊 Dataset Details

The model trains on a dataset containing student performance records. Key features include:

1. **Gender**: Male/Female
2. **Race/Ethnicity**: Groups A, B, C, D, E
3. **Parental Level of Education**: High school, Bachelor's, Master's, etc.
4. **Lunch**: Standard or Free/Reduced
5. **Test Preparation Course**: None or Completed
6. **Reading Score**: Numerical (0-100)
7. **Writing Score**: Numerical (0-100)

---

🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* Git

Installation

1. **Clone the repository**
```bash
git clone [https://github.com/your-username/student-score-prediction.git](https://github.com/your-username/student-score-prediction.git)
cd student-score-prediction

```


2. **Create a Virtual Environment**
```bash
conda create -p venv python=3.8 -y
conda activate venv/

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```


4. **Run the Application**
```bash
python app.py

```


5. **Access the App**
Open your browser and go to: `http://127.0.0.1:5000/`

---

🧠 Model Training Logic

The project uses a modular approach to model training:

1. **Data Ingestion**: Reads data from source (CSV/DB) and splits it into Train/Test sets.
2. **Data Transformation**: Handles missing values, performs One-Hot Encoding for categorical features, and Standard Scaling for numerical features using a `ColumnTransformer`.
3. **Model Selection**: The system iterates through multiple algorithms:
* Random Forest Regressor
* Decision Tree Regressor
* Gradient Boosting Regressor
* Linear Regression
* XGBoost Regressor
* CatBoost Regressor
* AdaBoost Regressor


4. **Hyperparameter Tuning**: Uses `GridSearchCV` (cv=5) to find the best parameters for each model.
5. **Evaluation**: Selects the model with the highest R2 Score (Accuracy).

**Current Best Model Performance:**

* **R2 Score**: ~88% (varies slightly based on tuning)

---

🤝 Contributing

Contributions are welcome!

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.


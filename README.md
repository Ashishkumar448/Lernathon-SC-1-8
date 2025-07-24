# 🚦 AI Food Adulteration Detection Dashboard

## 🌍 Why This Project?

Food adulteration is a major threat to public health, industry reputation, and consumer trust. Early and accurate detection helps save lives, money, and brand value.

---
## **Team Name:** TEAM(SC1)_8
## 👥 Team Members

| S.No. | Name                  | Roll Number        |
|-------|-----------------------|-------------------|
| 1     | Ashish Kumar Jena     | 24CSEDS018        |
| 2     | Kiran Surya Sahukar   | 24CSEAIML148      |
| 3     | Om Prasad Bartia      | 24CSEAIML122      |
| 4     | Dipak Kumar Behera    | 24CSEAIML247      |

---
[ppt link](learnathon.pptx)
---
## 🤖 What Does This App Do?

- **Predicts risk** (Minor/Moderate/Severe) for any new food sample.
- **Interactive dashboard:** Visualizes food safety, category risk, recent alerts, and performance trends.
- **Live data upload:** Instantly analyzes any compatible dataset via CSV upload.
- **Views for all roles:** Executive KPIs, Analyst exploration, Regulator compliance, and Operations monitoring.

---
![Lernathon-SC-1-8](https://socialify.git.ci/Ashishkumar448/Lernathon-SC-1-8/image?language=1&owner=1&name=1&stargazers=1&theme=Light)

## 🛠️ Key Features

|           |               |
|-----------|:-------------:|
| 🌀 **AI Prediction** | Classifies new samples via trained ML models |
| 📈 **Live Visuals** | KPIs, trend lines, category heatmaps, alert feeds |
| 📂 **Easy Data Upload** | Drop in new CSVs—dashboard adapts automatically |
| 🎛️ **Role-Based Tabs** | Executive, Analyst, Regulator, Operations views |
| 🤝 **No-Code** | Streamlit UI—intuitive, accessible anywhere |

---



## ⚙️ How It Works

Our Food Adulteration Detection System automates the entire workflow—from raw data to decision-ready analytics—using a robust data pipeline and scalable machine learning architecture.

---

### 🔄 Data Pipeline & Flowchart

**Workflow Steps:**
1. **Data Ingestion**  
   - Accepts CSV uploads directly via the web dashboard or loads existing data files.
2. **Preprocessing**  
   - Categorical encoding (brand, category, adulterant)
   - Parsing dates; extracting month and year  
   - Handling missing values, removing label-leaking columns
3. **Feature Engineering**  
   - Creation of time-related features
   - Synthesis of binary safety status fields
4. **Model Inference**
   - Data passed through a trained ML pipeline for severity prediction
5. **Analytics & Visualization**
   - Instantly updates all dashboard charts, KPIs, alert feeds, and stakeholder-specific panels
```
    +----------------+       +--------------+       +-----------------+       +-----------------+       +-------------------+
    |   Data Input   |  -->  | Preprocess & |  -->  |  Feature Engg.  |  -->  |   ML Model      |  -->  |  Dashboard &      |
    | (CSV Uploads)  |       |  Cleansing   |       |                 |       |  Inference      |       |  Visualization    |
    +----------------+       +--------------+       +-----------------+       +-----------------+       +-------------------+
```

---
```
                ┌──────────────────────┐
                │   User / Analyst     │
                └─────────┬────────────┘
                          │
         ┌────────────────▼────────────────┐
         │    Streamlit Web Application    │
         │  - Upload Data / Predict Sample │
         │  - Interactive Dashboards       │
         └────────────────┬───────────────┘
                          │
              +-----------▼-----------+
              |      Data Handler     |
              |  - Loads/filters CSV |
              +-----------┬-----------+
                          │
              +-----------▼-----------+
              |   Preprocessing &     |
              |   Feature Engineering |
              +-----------┬-----------+
                          │
         +----------------▼---------------+
         |     Trained ML Pipeline        |
         | (Preprocessing + Classifier)   |
         +----------------┬---------------+
                          │
            +-------------▼--------------+
            |     Severity Prediction    |
            |  & Dashboard Analytics    |
            +---------------------------+

```

**Summary:**  
- Users can upload data and input new sample attributes through the web UI.
- Every data entry or uploaded file flows through a uniform pipeline for cleaning, transformation, and prediction.
- The dashboard layer instantly reflects new insights for all roles, combining classic machine learning with real-time, user-driven analytics.
---
### 🤖 Machine Learning Model(s) Used

**Model Pipeline:**
- **Preprocessing:**  
  - One-hot encoder for categorical fields  
  - Standard/scaler when needed for numerics  
  - Drop columns that leak target
- **Estimator Choices:**  
  - **Random Forest Classifier (baseline)**
  - **XGBoost Classifier (optimized, selected via validation F1)**
- **Unified Pipeline:**  
  - Combines all steps (feature transforms + model) for seamless deployment
  - Supports both multiclass (severity) and binary (is_adulterated) tasks

**Typical Model Components:**

| Step                   | Type / Tool                             |
|------------------------|-----------------------------------------|
| Data Cleaning          | `pandas`, custom functions              |
| Feature Extraction     | `pandas`, `sklearn`                     |
| Categorical Encoding   | `OneHotEncoder`                         |
| Main Classifier        | `RandomForestClassifier`, `XGBClassifier`|
| Full Pipeline          | `sklearn.Pipeline`                      |
| Metrics                | F1, Accuracy, Confusion Matrix          |

---

### 🗺️ Architecture Diagram



## 📦 Dataset Description

### Dataset Sources

- Structured food audit data, laboratory adulteration records, and regulatory agency test results.

### Schema & Field Descriptions

| Field             | Description                                            |
|-------------------|-------------------------------------------------------|
| product_name      | Name of the food product                              |
| brand             | Brand or manufacturer name                            |
| category          | Food category/type (e.g., Dairy, Meat, Bakery)        |
| adulterant        | Detected contaminant or adulterant                    |
| severity          | Model-assessed severity: Minor/Moderate/Severe        |
| health_risk       | Health risk assessed from test                        |
| detection_date    | Date the product was tested                           |
| month/year        | Detection month and year, extracted                   |

### Sample Rows

| product_name   | brand   | category | adulterant            | severity | health_risk | detection_date | month | year |
|:-------------- |:------- |:-------- |:--------------------- |:-------- |:----------- |:-------------- |:----- |:-----|
| Milk Powder X  | BrandY  | Dairy    | Melamine              | Severe   | High        | 2025-07-12     | 7     | 2025 |
| Energy Bar     | BrandZ  | Bakery   | Artificial sweeteners | Moderate | Medium      | 2025-06-05     | 6     | 2025 |
| Cold Drink A   | BrandX  | Beverages| Water                 | Minor    | Low         | 2025-06-13     | 6     | 2025 |

---

## 🧠 Model Training and Evaluation

### Preprocessing Steps

- One-hot encoding for categorical fields (category, adulterant, brand)
- Date conversion and `month`, `year` extraction
- Removal of ID/leakage columns
- Handling missing data

### Training Workflow

1. **Data split** into training/validation/test sets.
2. **Pipeline** combining preprocessing and model (Random Forest / XGBoost).
3. **Model selection** and hyperparameter tuning for optimal F1 score.
4. **Serialization** using joblib for deployment.

### Model Performance Metrics

- Validation Macro F1 Score: ~0.78
- Test Macro F1 Score: ~0.74
- Confusion matrices and detailed per-class performance
- Best accuracy: Identifying “Severe” and “Safe” cases

---

## 🖥️ App/Dashboard Features

### Role-Based Dashboards

- **Executive:** At-a-glance KPIs (percent safe, critical cases), trends, traffic-light status
- **Analyst:** Heatmaps (category vs. severity), top adulterants by type, bar charts
- **Regulator:** Compliance per category, violation logs, severity distribution
- **Operations:** Weekly/monthly test volumes, critical alerts, sample processing status

### Input Forms

- **Sidebar:** Predicts risk for any product scenario you enter, instantly
- **CSV Uploader:** Instantly switches dashboard to the newly uploaded data

### Visualizations

- Pie charts for safety breakdown
- Line charts for time trends
- Category–severity heatmaps
- Bar/stacked bar charts for adulterant distributions
- Real-time critical alert feeds


  ### for executive
  <img width="1790" height="1181" alt="download" src="https://github.com/user-attachments/assets/c2ed57d7-09ae-43f5-8f32-4e6bea0b298a" />

  ### ANALYST DASHBOARD (For Data Scientists)

  <img width="1987" height="1769" alt="download (1)" src="https://github.com/user-attachments/assets/07683f6d-ce05-462e-9f38-00f9f6e031ee" />

  ### REGULATORY DASHBOARD (For Compliance Officers)
  <img width="1991" height="1181" alt="download (2)" src="https://github.com/user-attachments/assets/eb7c6194-924e-4ac5-9106-102142cdadca" />

  ### OPERATIONAL DASHBOARD (For Operations Teams)
   <img width="1790" height="1181" alt="download (3)" src="https://github.com/user-attachments/assets/5e12ffc7-6864-4232-bd6c-1961cf301c79" />  

### What-If & Scenario Analysis

- Each prediction or CSV upload immediately refreshes every chart, KPI, and dashboard panel for efficient role-based decision-making.

---

## 🚀 How to Run the Project

### Requirements

- Python 3.8 or newer
- Packages: `streamlit`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, `pyngrok`
- Browser access

### Step-by-Step Instructions

**On Colab:**
1. Upload your CSV and model files.
2. Install dependencies:
  ```bash
!pip install streamlit pandas scikit-learn xgboost matplotlib seaborn joblib pyngrok --quiet
  ```
3. Write the app file:
  ```python
with open("app.py", "w") as f:
f.write(app_code)
  ```
4. Run Streamlit + pyngrok:
  ``` python
from pyngrok import ngrok
import threading, time, os
def run_streamlit(): os.system('streamlit run app.py --server.port 8501')
threading.Thread(target=run_streamlit).start(); time.sleep(5)
print("Public URL:", ngrok.connect(8501))
  ```
5. Open the displayed ngrok URL in your browser.

**Locally:**
1. Place your CSV/model in the same folder as `app.py`.
2. Install Python dependencies.
3. Launch:
    ```pyhon
    streamlit run app.py
    ```

4. Open http://localhost:8501 in your browser.

### Sample Data & Model

- `food_adulteration_data.csv`
- Model pipeline: `xgb_pipe.pkl`

---

## 🛠️ Using the Prediction App

### 🔮 Predicting a New Sample

- Use the **sidebar form** to enter details for any food sample:
    - Product name, brand, category, adulterant, health risk, detection month and year.
- Click **Predict**.
- The app will instantly display:
    - The predicted severity label (**Minor, Moderate, or Severe**)
    - The confidence score for the prediction (in %)
- The new sample, along with its predicted risk, is added to all dashboard visualizations in real time for immediate impact analysis.

### 🆕 Adding/Uploading New Food Data

- Under **"Upload New Data File (Optional)"** in the sidebar, click **Browse files** or drag and drop your own CSV.
- The app will automatically use the uploaded data for:
    - All analytics and dashboard charts
    - All future predictions
    - Filtering, role-specific tabs, and alerts
- No code changes, app restarts, or manual steps are needed—data uploads are seamless.

### 🔄 How Uploaded Data is Used

- The uploaded CSV **overrides the default dataset** for the entire session.
- All dashboard features (KPIs, time trends, heatmaps, scenario analysis) and predictions will reflect your uploaded data instantly.
- You can experiment with testing batches, regulatory reporting, or simulation scenarios by simply uploading different files.
- To revert to the default dataset, just **refresh the page** and skip the upload.

---

## 📊 Key Insights & Results

### Data Exploration Highlights
- **Most common adulterants:** Melamine and Artificial Sweeteners were top contaminants in Dairy and Bakery items, respectively.
- **Risk trends:** Severe cases have seasonal spikes—especially mid-year—coinciding with increased production periods.
- **Category exposure:** Dairy and Beverages showed both the highest testing volume and the highest "Severe" rate.

### Model Insights
- **Top features:** Category, adulterant type, and health risk had the strongest influence on severity prediction.
- **Confusion matrix:** Model accurately distinguishes severe risks; rare confusion between "Minor" and "Severe".
- **Robust to new data:** Retraining with new CSV uploads showed stable performance across multiple batches.

### Impact Examples
- **Early detection:** Predicting new samples before market release, allowing recall/pre-emptive action.
- **Speed:** Dashboard reduces analytics turnaround for regulators from hours to minutes.
- **Transparency:** Executive/Operations can see the latest risks and alerts without waiting for manual reports.

---

## ⚠️ Limitations & Future Work
### Current Known Issues
- Predictions are only as accurate as the training data's quality and scope.
- Out-of-domain data (new categories or never-seen adulterants) may not be handled optimally.
- Requires users to supply correctly-formatted CSV for seamless upload.

### Next Steps/Enhancements Proposed
- Integrate automatic retraining and drift monitoring with regular new data.
- Add user authentication for report customization and data privacy.
- Support more granular chemical/ingredient analysis and image-based detection.
- Real-time alert email/SMS integration for critical cases.

---


## 🙏 Acknowledgements

- **Data Sources:** Regulatory agencies, open food safety databases, simulated records.
- **Major Libraries/Frameworks:** Streamlit, scikit-learn, pandas, XGBoost, matplotlib, seaborn, pyngrok
- **Special Thanks:** Mentor/Support, and all contributors!

---

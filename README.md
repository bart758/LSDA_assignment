[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Ky09GFhe)
# Forecasting Wind Power Production in Orkney

The Orkney archipelago in the UK generates more than its net electricity needs from renewable energy, primarily wind power. However, wind power production is inherently variable and difficult to predict. Accurate forecasting is critical for grid stability, planning, and efficient energy distribution.

In this assignment, you will design, implement, and serve a **reproducible machine learning pipeline** that predicts wind power generation in Orkney using real-world historical power data and weather forecasts. The emphasis is not on achieving the best possible accuracy, but on demonstrating good practices in **data engineering, modeling, reproducibility, and deployment** across the full machine learning lifecycle.

To get started with this project, click `<> Code` and then start a codespace on main.

In the assignment repository you will find a Jupyter Notebook `exploration.ipynb` with detailed instructions and helper code to get you started. Go through the steps in this notebook to get familiar with the datasets, pipelines, and working with MLflow. Your final project should solve the tasks below.

---


## Task description

You will build a **training and deployment pipeline** that transforms raw data into a forecasting service.
  
  - [ ] Data alignment

  * Load power generation and weather forecast data
  * Align the two data sources temporally (e.g. resampling, joins, or interpolation)
  * Clearly justify your alignment strategy and discuss its implications

  - [ ] Data preprocessing with pipelines

   * Apply a suitable data-splitting strategy for train and test
   * Handle missing values
   * Transform wind direction into a numeric representation (e.g. encoding, radians, vector form)
   * Scale numerical features where appropriate


  - [ ] Model training and evaluation

  * Train at least 2 regression models of your choice
  * Use evaluation metrics appropriate for regression
  * Use the `future.csv` file to generate future predictions (to check that your model works on new data)

  - [ ] Experiment tracking with MLflow
  
  * Log parameters, metrics, and artifacts using MLflow Tracking
  * Organize experiments and runs clearly
  * Use MLflow to compare model variants and select a best model
  
  - [ ] Model serving
  
  * Register the selected model using the MLflow Model format
  * Serve the model
  * Expose a prediction endpoint that accepts weather inputs and returns power forecasts
    
  - [ ] Reproducibility with MLflow Projects
  
  * Package your training code as an MLProject
  * Specify dependencies using an environment file
  * Ensure the project can be executed from scratch on another machine
  
  - [ ] Reflection
  
  * Discussion of training data window size (e.g. 90 days)
  * Mention limitations of your approach and potential improvements
---

## Deliverables

### Report (PDF)

In the report you briefly describe how you solved the tasks above. For each task:

- explain how you solved it and why you chose a particular approach
- show snippets of your code

Include your results with screen captures from the MLflow UI.

**Maximum length:** 5 pages (excluding figures)

**File name:** `<itu_username>-A1-report.pdf`

**Submission:** on LearnIt

**Include the link to the Git repository in the report.**

### Code repository

* A Git repository containing all code needed to run the pipeline
* Code must run without errors
* Code must be packaged as MLProject; it should be possible to run it with `mlflow run`
* Include instructions for running training and deployment

---

## Assessment criteria

Your submission will be evaluated based on:

* Pipeline design and correctness
* Quality of preprocessing, feature engineering, modeling, and evaluation decisions
* Appropriate use of MLflow for tracking and reproducibility
* Clarity and depth of reflection in the report

Model accuracy alone is *not* a grading criterion.

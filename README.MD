# [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) Prediction

This project implements a complete Machine Learning pipeline to predict the number of bikes rented in Seoul based on weather conditions and time factors.

## 🚀 Key Features
- **Full Pipeline:** From raw data cleaning to model evaluation.
- **Multi-Model Comparison:** Compares Linear Regression, Random Forest, Gradient Boosting, Decision Tree, and KNN.
- **Feature Engineering:** Includes cyclical encoding for temporal features and multicollinearity handling.
- **Visual Diagnostics:** Automatically generates correlation heatmaps, feature importance plots, and error analysis charts.

## 🛠 Project Structure
- `main.py`: The entry point to run the application.
- `starter/`: Orchestrates the ML workflow.
- `model/`: Contains logic for data processing, normalization, and prediction.
- `util/`: Configuration constants and utility helpers.
- `source_data/`: Folder for the `SeoulBikeData.csv` dataset.
- `plots/`: Automatically generated visualizations.

## 📦 Requirements
- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn

## 🏃 How to Run
1. **Prepare Data:** Ensure your dataset is located in `source_data/SeoulBikeData.csv`.
2. **Execute:** Run the main script from the root directory:
   ```bash
   python main.py
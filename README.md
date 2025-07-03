🔧 Goal:

Predict the power consumption of a digital design (like a processor block or a datapath) using high-level RTL features.

📌 Why Important in VLSI:

Early prediction of power helps reduce simulation time and avoids over-design. It speeds up the design cycle and helps in power-performance-area (PPA) optimization.


#  VLSI Power Consumption Prediction (ML + Streamlit)

This project predicts the power consumption (in milliwatts) of VLSI RTL designs using Machine Learning.  
It uses `RandomForestRegressor` trained on RTL-level features such as:

- Number of Gates
- Frequency (MHz)
- Area (mm²)
- Pipeline Stages

##  Features

- Upload your own `.csv` dataset
- Interactive sliders for prediction
- Real-time power estimation
- Visual feedback (low / medium / high power)
- Feature importance chart

- ##  Installation

```bash
git clone https://github.com/your-username/VLSI_Power_Predictor.git
cd VLSI_Power_Predictor
pip install -r requirements.txt

# ⚡ EV Sales Forecasting App

A Streamlit-based web app that predicts future Electric Vehicle (EV) sales for Indian states using machine learning. Built with real-world datasets and provides insights into EV market trends. **Now includes an offline Power BI dashboard for detailed visual analysis!**

🚀 **Live Demo**  
Check out the live app here: [EV Forecasting App](#) *(update link if hosted)*

📌 **Features**
✅ Predict future EV sales for Indian companies  
✅ Inputs include average price, revenue, vehicles sold, lag features, rolling averages, etc.  
✅ Interactive and user-friendly interface built with Streamlit  
✅ Machine Learning model trained on historical EV sales data  
✅ Integrated offline Power BI dashboard for deep visual insights  
✅ Ready for demonstration in CVs, interviews, and data analytics portfolios

🛠 **Technologies Used**
- Python
- Pandas, NumPy
- Scikit-Learn, XGBoost
- Streamlit
- Power BI

📂 **Project Structure**
├── app.py # Streamlit app main file
├── requirements.txt # Python dependencies
├── ev_model.pkl # Trained ML model
└── powerbi_dashboard/
├── ev_dash.pbix # Power BI dashboard file
└── ev_dash.pdf # Power BI dashboard PDF export

## 🚨 How to Run Locally


🚨 **How to Run Locally**
Clone the repository:
git clone https://github.com/aastha2923/ev-forecasting-app.git
cd ev-forecasting-app

## Install dependencies:
pip install -r requirements.txt

## Run the app:
streamlit run app.py


📈 **Power BI Dashboard**
The `powerbi_dashboard/` folder contains:
- **ev_dash.pbix**: Editable Power BI dashboard file.
- **ev_dash.pdf**: Exported PDF of the dashboard for quick viewing.
Use these to explore EV sales data visually and include them in your portfolio or presentations.

📝 **Author**
Aastha Choubey

📃 **License**
This project is licensed under the MIT License.

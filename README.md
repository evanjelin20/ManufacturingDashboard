# Manufacturing Quality & Production Insights

A small Streamlit portfolio prototype for a manufacturing/production-company interview.

## Features
- Production output, estimated quality pass rate, and average downtime KPIs
- Filters by machine and shift
- Production trend, defect-rate comparison, and shift-share charts
- Random Forest quality-risk demo
- Downloadable sample CSV

## Important data note
The app creates synthetic sample data and demo labels so it can run immediately without external files. It is not connected to a real factory. The ML model's score is only a result on generated data and must not be presented as real-world validated accuracy.

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Run: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

## Deploy with Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Open https://share.streamlit.io/
3. Sign in with GitHub, choose the repository, select `app.py`, and deploy.
4. Test the public URL before sharing it.

## Resume wording (use only after you have run and tested it)
**Manufacturing Quality & Production Insights Dashboard | Python, Streamlit, Scikit-learn**
- Built an interactive dashboard to monitor production output, downtime, and machine-level defect trends using synthetic manufacturing data.
- Implemented a Random Forest-based quality-risk demo and interactive visualizations, with a downloadable CSV report.

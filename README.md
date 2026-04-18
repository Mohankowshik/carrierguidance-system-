# Career Guidance System

A comprehensive web-based application designed to help students discover suitable career paths based on their academic performance and technical interests.

## 🚀 Features

- **Academic Career Analysis**: Recommends careers based on subject marks (Math, Physics, Chemistry, etc.) using a Random Forest machine learning model.
- **Interest-Based Analysis**: Suggests careers using a weighted scoring system:
  - **65% Technical Interests**: Based on user-selected tech skills (AI, Web Dev, Cyber Security, etc.).
  - **35% Holland Personality Type**: Based on the RIASEC (Realistic, Investigative, Artistic, Social, Enterprising, Conventional) model.
- **Comprehensive Analysis**: Combines both academic and interest data for a holistic career recommendation.
- **Visual Insights**: Generates dynamic pie charts showing the top 5 recommended careers for each analysis.
- **Responsive UI**: Modern, user-friendly interface built with Bootstrap 5.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib (with Base64 encoding for web display)
- **Frontend**: HTML5, CSS3, Jinja2, Bootstrap 5

## 📁 Project Structure

- `app.py`: The main Flask application handling routing and recommendation logic.
- `train_model.py`: Script to train the Random Forest model and save it.
- `Models/`: Directory containing the trained `model.pkl` and `scaler.pkl`.
- `templates/`: Jinja2 HTML templates for the web interface.
- `student-scores.csv`: The dataset used for training the model.
- `requirements.txt`: List of required Python packages.

## ⚙️ Installation & Setup

1. **Clone the repository** (or extract the project files).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model** (if `model.pkl` is missing):
   ```bash
   python train_model.py
   ```
4. **Run the application**:
   ```bash
   python app.py
   ```
5. **Access the system**: Open your browser and navigate to `http://127.0.0.1:5000/`.

## 📊 How it Works

1. **Academic Analysis**: The system takes student details and subject scores, scales them, and uses a Random Forest Classifier to predict the most likely career aspirations based on historical data.
2. **Interest Analysis**:
   - **Technical Mapping**: Each selected technical interest adds weight to specific career paths.
   - **Holland Code**: The personality test results are mapped to corresponding professional categories.
   - **Weighted Score**: Final recommendations are calculated using the 65/35 weighted formula.
3. **Visualization**: Matplotlib generates charts on-the-fly, which are converted to Base64 strings and rendered directly in the browser without requiring temporary image files.

## ⚠️ Notes

- The system is designed to be robust; if `matplotlib` is not installed, it will still provide text-based recommendations but skip the chart generation.
- Ensure the `Models/` folder contains both `model.pkl` and `scaler.pkl` for the academic section to function.

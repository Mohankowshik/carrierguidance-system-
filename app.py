from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import numpy as np
import io
import base64
import os
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

app = Flask(__name__)

# Load models and scaler
MODEL_PATH = "Models/model.pkl"
SCALER_PATH = "Models/scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    class_names = model.classes_
else:
    model = None
    scaler = None
    class_names = []

# Personality test data (Holland Code)
holland_questions = {
    "R": ["I enjoy working with machines and tools.", "I like practical tasks.", "I prefer building things."],
    "I": ["I enjoy solving puzzles.", "I like experiments.", "I enjoy analyzing data."],
    "A": ["I enjoy drawing.", "I like expressing through music.", "I enjoy writing."],
    "S": ["I enjoy helping people.", "I like volunteering.", "I enjoy teaching."],
    "E": ["I enjoy leading teams.", "I like persuading others.", "I enjoy organizing."],
    "C": ["I prefer working with numbers.", "I like organized systems.", "I enjoy record-keeping."]
}

# Mapping Holland Types to all model careers for scoring
holland_to_careers = {
    "R": ["Construction Engineer", "Mechanical Engineer", "Game Developer", "Doctor"],
    "I": ["Scientist", "Software Engineer", "Doctor", "Data Analyst", "Researcher"],
    "A": ["Artist", "Designer", "Writer", "Game Developer"],
    "S": ["Teacher", "Doctor", "Lawyer", "Nurse", "Counselor"],
    "E": ["Business Owner", "Entrepreneur", "Stock Investor", "Real Estate Developer", "Lawyer", "Manager", "Salesperson"],
    "C": ["Accountant", "Banker", "Data Analyst", "Government Officer", "Stock Investor"]
}

# Technical interests mapping to model careers
technical_interests_mapping = {
    "Coding & Programming": ["Software Engineer", "Game Developer"],
    "Data Analysis": ["Scientist", "Accountant", "Banker", "Software Engineer"],
    "Web Development": ["Software Engineer", "Designer"],
    "Mobile App Development": ["Software Engineer", "Game Developer"],
    "Artificial Intelligence": ["Software Engineer", "Scientist"],
    "Cyber Security": ["Software Engineer", "Government Officer"],
    "Cloud Computing": ["Software Engineer"],
    "Database Management": ["Software Engineer", "Accountant", "Banker"],
    "UI/UX Design": ["Designer", "Software Engineer"],
    "Robotics & Automation": ["Software Engineer", "Construction Engineer", "Scientist"],
    "Game Design": ["Game Developer", "Designer", "Artist"],
    "Structural Engineering": ["Construction Engineer"],
    "Medical Research": ["Doctor", "Scientist"],
    "Financial Modeling": ["Stock Investor", "Banker", "Accountant"],
    "Legal Research": ["Lawyer", "Government Officer"],
    "Network Engineering": ["Software Engineer"],
    "Software Testing & QA": ["Software Engineer"],
    "Real Estate Analysis": ["Real Estate Developer", "Stock Investor"],
    "Strategic Planning": ["Business Owner", "Government Officer", "Stock Investor"],
    "Scientific Experimentation": ["Scientist", "Doctor"],
    "Mathematical Logic": ["Scientist", "Software Engineer", "Banker", "Accountant"],
    "System Administration": ["Software Engineer", "Government Officer"],
    "Blockchain Technology": ["Software Engineer", "Stock Investor", "Banker"],
    "Machine Learning": ["Software Engineer", "Scientist"],
    "Electronics & Hardware": ["Construction Engineer", "Software Engineer"]
}

personality_info = {
    "R": {"name": "Realistic", "description": "Practical, hands-on, likes working with tools and machines."},
    "I": {"name": "Investigative", "description": "Analytical, problem solver, likes science and math."},
    "A": {"name": "Artistic", "description": "Creative, expressive, likes music, art, and writing."},
    "S": {"name": "Social", "description": "Helpful, nurturing, likes teaching and helping others."},
    "E": {"name": "Enterprising", "description": "Leader, persuasive, likes business and managing people."},
    "C": {"name": "Conventional", "description": "Organized, structured, likes numbers and systems."}
}

# Helper function to generate a pie chart
def generate_pie_chart(labels, values, title):
    if not has_matplotlib:
        return ""
    try:
        plt.figure(figsize=(8, 6))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(title)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        return base64.b64encode(img_buffer.read()).decode('utf-8')
    except Exception:
        return ""

# Career recommendation function
def get_career_recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                              weekly_self_study_hours, math_score, history_score, physics_score,
                              chemistry_score, biology_score, english_score, geography_score,
                              total_score, average_score):
    if not model or not scaler:
        return [("Model not loaded", 0)], ""

    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                             weekly_self_study_hours, math_score, history_score, physics_score,
                             chemistry_score, biology_score, english_score, geography_score, total_score,
                             average_score]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Get top three predicted classes along with their probabilities
    top_indices = np.argsort(-probabilities[0])
    top_3_idx = top_indices[:3]
    top_3_recommendations = [(class_names[idx], round(probabilities[0][idx] * 100, 2)) for idx in top_3_idx]
    
    # Get top 5 for pie chart
    top_5_idx = top_indices[:5]
    top_5_labels = [class_names[idx] for idx in top_5_idx]
    top_5_values = [probabilities[0][idx] for idx in top_5_idx]
    
    chart_base64 = generate_pie_chart(top_5_labels, top_5_values, "Top 5 Recommended Careers (Academic)")

    return top_3_recommendations, chart_base64

# Personality test processing
def process_personality_test(data):
    # 1. Holland Personality Score (35% weight)
    holland_scores = {ptype: 0 for ptype in holland_questions}
    for key, value in data.items():
        if '_' in key:
            ptype = key.split('_')[0]
            if ptype in holland_scores:
                try:
                    holland_scores[ptype] += int(value)
                except ValueError:
                    continue
    
    # Normalize Holland scores (max possible score per category is 15 if 3 questions * 5)
    dominant_type = max(holland_scores, key=holland_scores.get)
    
    # 2. Technical Interest Score (65% weight)
    selected_interests = data.get('technical_interests', [])
    if isinstance(selected_interests, str):
        selected_interests = [selected_interests]
        
    # Initialize career scores
    # We'll use all careers from the model classes
    career_scores = {career: 0.0 for career in class_names}
    
    # Score from Technical Interests (65%)
    if selected_interests:
        tech_weight_per_interest = 65.0 / len(selected_interests)
        for interest in selected_interests:
            if interest in technical_interests_mapping:
                matched_careers = technical_interests_mapping[interest]
                for career in matched_careers:
                    if career in career_scores:
                        career_scores[career] += tech_weight_per_interest
    
    # Score from Holland Personality (35%)
    # Distribute 35% among careers associated with the Holland types based on their scores
    total_holland_points = sum(holland_scores.values())
    if total_holland_points > 0:
        for ptype, score in holland_scores.items():
            if score > 0:
                ptype_weight = (score / total_holland_points) * 35.0
                associated_careers = holland_to_careers.get(ptype, [])
                if associated_careers:
                    weight_per_career = ptype_weight / len(associated_careers)
                    for career in associated_careers:
                        if career in career_scores:
                            career_scores[career] += weight_per_career

    # Sort careers by score
    sorted_careers = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 3 and top 5
    top_3_careers = [(name, round(score, 2)) for name, score in sorted_careers[:3]]
    top_5_careers = sorted_careers[:5]
    
    # Filter out zero-score careers for pie chart if necessary, but keep top 5
    chart_labels = [c[0] for c in top_5_careers if c[1] > 0]
    chart_values = [c[1] for c in top_5_careers if c[1] > 0]
    
    # If no scores, provide some defaults to avoid empty pie chart
    if not chart_values:
        chart_labels = ["No data"]
        chart_values = [1]

    chart_base64 = generate_pie_chart(chart_labels, chart_values, "Top 5 Recommended Careers (Interests)")

    return {
        "personality": personality_info[dominant_type]["name"],
        "description": personality_info[dominant_type]["description"],
        "recommendations": top_3_careers,
        "chart": chart_base64,
        "scores": holland_scores
    }


# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/career', methods=['GET', 'POST'])
def career():
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            part_time_job = request.form.get('part_time_job', 'false') == 'true'
            absence_days = int(request.form['absence_days'])
            extracurricular_activities = request.form.get('extracurricular_activities', 'false') == 'true'
            weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
            math_score = int(request.form['math_score'])
            history_score = int(request.form['history_score'])
            physics_score = int(request.form['physics_score'])
            chemistry_score = int(request.form['chemistry_score'])
            biology_score = int(request.form['biology_score'])
            english_score = int(request.form['english_score'])
            geography_score = int(request.form['geography_score'])
            total_score = float(request.form['total_score'])
            average_score = float(request.form['average_score'])

            recommendations, chart = get_career_recommendations(
                gender, part_time_job, absence_days, extracurricular_activities,
                weekly_self_study_hours, math_score, history_score, physics_score,
                chemistry_score, biology_score, english_score, geography_score,
                total_score, average_score
            )

            return render_template('results.html', recommendations=recommendations, chart=chart)
        except Exception as e:
            return f"Error processing request: {str(e)}", 400

    return render_template('career.html')

@app.route('/personality', methods=['GET', 'POST'])
def personality():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data received"}), 400

            results = process_personality_test(data)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('personality.html', questions=holland_questions, technical_interests=technical_interests_mapping.keys())

@app.route('/combined', methods=['GET', 'POST'])
def combined():
    if request.method == 'POST':
        try:
            # Process career form
            gender = request.form['gender']
            part_time_job = request.form.get('part_time_job', 'false') == 'true'
            absence_days = int(request.form['absence_days'])
            extracurricular_activities = request.form.get('extracurricular_activities', 'false') == 'true'
            weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
            math_score = int(request.form['math_score'])
            history_score = int(request.form['history_score'])
            physics_score = int(request.form['physics_score'])
            chemistry_score = int(request.form['chemistry_score'])
            biology_score = int(request.form['biology_score'])
            english_score = int(request.form['english_score'])
            geography_score = int(request.form['geography_score'])
            total_score = float(request.form['total_score'])
            average_score = float(request.form['average_score'])

            career_results, career_chart = get_career_recommendations(
                gender, part_time_job, absence_days, extracurricular_activities,
                weekly_self_study_hours, math_score, history_score, physics_score,
                chemistry_score, biology_score, english_score, geography_score,
                total_score, average_score
            )

            # Process personality test and technical interests
            personality_data = {}
            # Technical interests come as a list from combined form
            personality_data['technical_interests'] = request.form.getlist('technical_interests')
            
            for key in request.form:
                if any(key.startswith(k) for k in holland_questions.keys()):
                    try:
                        personality_data[key] = int(request.form[key])
                    except ValueError:
                        continue

            personality_results = process_personality_test(personality_data)

            return render_template('combined_results.html',
                                career_results=career_results,
                                career_chart=career_chart,
                                personality_results=personality_results)
        except Exception as e:
            return f"Error processing request: {str(e)}", 400

    return render_template('combined.html', questions=holland_questions, technical_interests=technical_interests_mapping.keys())


if __name__ == '__main__':
    app.run(debug=True)

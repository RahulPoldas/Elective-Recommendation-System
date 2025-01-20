from flask import Flask, request, render_template, jsonify
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Load the dataset
file_path = 'Anon Data.txt'
df = pd.read_csv(file_path, delimiter='\t')

# Prepare and train the model
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(df[['RollNumber', 'Course Code', 'Grade Points']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd = SVD()
svd.fit(trainset)
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
rating_scale_range = 10
accuracy_normalized = 1 - (rmse / rating_scale_range)
print(f"Model Accuracy (normalized): {accuracy_normalized:.2f}")
print("RMSE: ", rmse)

# Helper function for recommendations
def recommend_courses(student_id, df, model, top_n=1):
    all_courses = df['Course Code'].unique()
    rated_courses = df[df['RollNumber'] == student_id]['Course Code'].values
    unrated_courses = [course for course in all_courses if course not in rated_courses]
    predicted_ratings = [(course, model.predict(str(student_id), str(course)).est) for course in unrated_courses]
    top_courses = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"course": course, "predicted_grade": round(rating, 2)} for course, rating in top_courses]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Extract form data
    roll_index = int(request.form['roll_index'])
    student_name = request.form['student_name']
    ggpa = float(request.form['ggpa'])
    section = request.form['section']
    course1 = request.form['course']
    course2 = request.form['course2']
    course3 = request.form['course3']

    # Validate roll index
    if roll_index < 0 or roll_index >= len(df):
        return jsonify({"error": "Invalid roll index"}), 400
    
    # Get student ID based on roll index
    student_id = df['RollNumber'].iloc[roll_index]

    # Get recommendations
    recommendations = recommend_courses(student_id, df, svd, top_n=1)

    # Render the template with user data and recommendations
    return render_template(
        'recommendation.html',
        student_id=student_id,
        student_name=student_name,
        ggpa=ggpa,
        section=section,
        selected_courses=[course1, course2, course3],
        recommendations=recommendations
    )

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, render_template, jsonify
import pandas as pd
from surprise import Dataset, Reader, SVD , accuracy
from surprise.model_selection import train_test_split

app = Flask(__name__)

file_path = 'Anon Data.txt'
df = pd.read_csv(file_path, delimiter='\t')

reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(df[['RollNumber', 'Course Code', 'Grade Points']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd = SVD()
svd.fit(trainset)
predictions=svd.test(testset)
rmse=accuracy.rmse(predictions)

def recommend_courses(student_id, df, model, top_n=1):
    all_courses = df['Course Code'].unique()
    rated_courses = df[df['RollNumber'] == student_id]['Course Code'].values
    unrated_courses = [course for course in all_courses if course not in rated_courses]
    predicted_ratings = [(course, model.predict(str(student_id), str(course)).est) for course in unrated_courses]
    top_courses = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"course": course, "predicted_grade": round(rating, 2), "accuracy": rmse} for course, rating in top_courses]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    roll_index = int(request.form['roll_index'])
    if roll_index < 0 or roll_index >= len(df):
        return jsonify({"error": "Invalid roll index"}), 400
    
    student_id = df['RollNumber'].iloc[roll_index]
    recommendations = recommend_courses(student_id, df, svd, top_n=1)
    return render_template('recommendation.html', student_id=student_id, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect
from object_detector import get_object_detection, load_object_detector
from object_classifier import get_image_clasess, load_object_classifier, get_class_labels, generate_comment
import os
from pathlib import Path

#load models and labels
labels = get_class_labels()
model_object_detector = load_object_detector()
model_object_classifier = load_object_classifier(labels)

app = Flask(__name__)
UPLOAD_FOLDER = Path('static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect('/')
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect('/')

    if file:
        # Save the uploaded file to the uploads folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        # Render the index template with the uploaded file name

        result_df = get_object_detection(model_object_detector, Path(os.path.join("static/uploads", file.filename)))
        object_dict = get_image_clasess(model_object_classifier, labels, result_df["object_list"])
        comments = generate_comment(result_df["person_count"], object_dict)
        print("comments : ", comments)
        return render_template('index.html', comments=comments, filename=file.filename)    

# app.run(port=5008, debug=True)
# if __name__ == '__main__':
    # if not os.path.exists('uploads'):
    #     os.makedirs('uploads')
    # app.run(debug=True)
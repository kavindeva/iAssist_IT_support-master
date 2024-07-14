import os
# import time
# import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "C:\\inetpub\\wwwroot\\iAssist_IT_support\\New IT support datasets"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/file-upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        response = jsonify({'message': 'No file part in the request'})
        response.status_code = 400
        return response
    file = request.files['file']
    if file.filename == '':
        response = jsonify({'message': 'No file selected for uploading'})
        response.status_code = 400
        return response
    if file and allowed_file(file.filename):
        print("if working")
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(filename)
        print(file)
        response = jsonify({'message': 'Spreadsheet uploaded successfully'})
        response.status_code = 201
        return response
    else:
        response = jsonify({'message': 'Allowed file types are csv or xlsx'})
        response.status_code = 400
        return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

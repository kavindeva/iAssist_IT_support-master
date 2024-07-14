import os
import time
import glob
# import json
import pandas as pd
from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socket = SocketIO(app, cors_allowed_origins="*")
# socket1 = CORS

app.config['UPLOAD_FOLDER'] = "C:\\inetpub\\wwwroot\\iAssist_IT_support\\New IT support datasets"
success = False


@socket.on('connect')
def connected():
    print('connect')


@socket.on('message')
def handle_message(data):
    global success
    print("Received filename: ", data["data"])
    filenames = data["data"]
    print(filenames)
    folderpath = glob.glob('C:\\inetpub\\wwwroot\\iAssist_IT_support\\New IT support datasets\\*.csv')
    latest_file = max(folderpath, key=os.path.getctime)
    print(latest_file)
    time.sleep(3)
    if filenames in latest_file:
        df1 = pd.read_csv("C:\\inetpub\\wwwroot\\iAssist_IT_support\\New IT support datasets\\" +
                          filenames, names=["errors", "solutions"])
        df2 = pd.read_csv("C:\\inetpub\\wwwroot\\iAssist_IT_support\\existing_tickets.csv",
                          names=["errors", "solutions"])
        combined_csv = pd.concat([df2, df1])
        combined_csv.to_csv("C:\\inetpub\\wwwroot\\iAssist_IT_support\\new_tickets-chatdataset.csv",
                            index=False, encoding='utf-8-sig')
        time.sleep(3)
        success = not success
        print(success)
        socket.emit('receiver', {'Success1': 'New data merged with existing datasets'})
        if success:
            print(os.getcwd())
            socket.emit('receiver', {'Success2': "New model training is in progress don't upload new file"})
            command = "python main.py"
            os.system(command)
            socket.emit('receiver', {'Success3': "Model created successfully for sent file %s" % filenames})
    else:
        socket.emit('receiver', "No any progress")


if __name__ == "__main__":
    socket.run(app, host="0.0.0.0", port=8085)

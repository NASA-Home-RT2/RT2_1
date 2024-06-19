from flask import Flask,jsonify,request
from flask_cors import CORS
from datetime import datetime
from flask_compress import Compress
import json

compress = Compress()

app = Flask(__name__, static_folder='server_data/static', static_url_path='')
compress.init_app(app)
CORS(app)
@app.route('/')
def index():
    return app.send_static_file('templates/index.html')

@app.route('/index.html')
def index2():
    return app.send_static_file('templates/index.html')

@app.route('/status.html')
def status():
    return app.send_static_file('templates/status.html')

@app.route('/calendar.html')
def calendar():
    return app.send_static_file('templates/calendar.html') 

@app.route('/loading.html')
def loading():
    return app.send_static_file('templates/loading.html') 

@app.route('/inventory.html')
def inventory():
    return app.send_static_file('templates/inventory.html')

@app.route('/xx.html')
def xx():
    return app.send_static_file('templates/xx.html')

@app.route('/api/data', methods = ['GET'])
def ReturnData():
    if(request.method == 'GET'):
        with open ('./server_data/data/data.json','r') as file:
            data=json.load(file)
        return jsonify(data)

@app.route('/api/dataCalendar', methods = ['GET'])
def ReturnDataCalendar():
    if(request.method == 'GET'):
        with open ('./server_data/data/dataCalendar.json','r') as file:
            data=json.load(file)
        return jsonify(data)

@app.route('/api/dataInventory', methods = ['GET'])
def ReturnDataInventory():
    if(request.method == 'GET'):
        with open ('./server_data/data/inventory.json','r') as file:
            data=json.load(file)
        return jsonify(data)
    

@app.route('/api/dataActualization', methods = ['GET'])
def ReturnDataAct():
    if(request.method == 'GET'):
        with open ('./server_data/data/responseClient.json','r') as file:
            data=json.load(file)
        return jsonify(data)

@app.route('/update', methods=['POST'])
def updates():
     if(request.method == 'POST'):
        time=str(datetime.now())
        data = {"date": time }
        
        with open('./server_data/data/responseServer.json', 'w') as file:
            json.dump(data, file)
            return "complete "


if __name__=='__main__':
    app.run(debug=True, port=3000)
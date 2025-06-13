from flask import Flask, request
from inference import AppModel

app = Flask(__name__)
model = AppModel()

@app.route('/get_category', methods=['GET'])
def print_message():
    message = request.args.get('message', default='', type=str)   
    model_type = request.args.get('model_type', default='small', type=str) 
    return model.inference(message, model_type)

if __name__ == '__main__':
    app.run()
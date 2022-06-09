from flask import Flask, request, jsonify
from resume_generator import generate_ai_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello():
    return 'Hello, Flask!'


@app.route('/text/new', methods=['POST'])
def get_ai_text():
    print('----------------------------------------------------------------')
    print(request.headers)
    print(request.get_json())
    print('----------------------------------------------------------------')

    params = request.get_json()
    user_text = params['user_text']
    ai_text = generate_ai_text(user_text)
    res = jsonify(ai_text=ai_text)
    return res


if __name__ == '__main__':
    app.run()

from flask import Flask, request, jsonify
from resume_generator import generate_ai_text

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, Flask!'


@app.route('/text/new', methods=['POST'])
def get_ai_text():
    params = request.get_json()
    user_text = params['user_text']
    ai_text = generate_ai_text(user_text)
    return jsonify(
        ai_text=ai_text
    )


if __name__ == '__main__':
    app.run()

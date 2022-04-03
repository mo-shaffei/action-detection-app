from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/logs')
def logs():
    return render_template('logs.html')


@app.route('/connect')
def connect():
    print("test")
    return "test"


if __name__ == '__main__':
    app.run(debug=True)

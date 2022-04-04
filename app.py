from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
import threading
import logic

app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('homepage.html')


@app.route('/logs')
def logs():
    return render_template('logs.html')


@app.route('/connect')
def connect():
    t = threading.Thread(target=logic.connect_thread)  # create new thread
    t.setDaemon(True)
    t.start()  # start thread
    return Response(status=204)


if __name__ == '__main__':
    app.run(debug=True)

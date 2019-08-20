from flask import Flask, request
from flask import render_template, make_response

app = Flask(__name__)

@app.route('/')
def index():
    return 'Index page'

@app.route('/hello')
def hello():
    return 'Hello!'

@app.route('/user/<username>')
def show_user_profile(uesrname):
    return f'User {uesrname}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f'Post {post_id}'

@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'

@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['usesrname'])
        else:
            error = 'Invalid username/password'
        return render_template('login.html', error=error)

@app.errorhandler(404)
def not_found(error):
    response = make_response(render_template('error.html'), 404)
    response.headers['X-Something'] = 'A value'
    return response



if __name__ == "__main__":
    app.debug = True
    app.run()

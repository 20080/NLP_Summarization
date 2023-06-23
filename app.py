from flask import Flask, render_template
import pandas as pd
from flask_mysqldb import MySQL

import nlp

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'database-1.cx22qud2cpdi.eu-north-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '20080114040'
app.config['MYSQL_DB'] = 'NLPFLASK'

mysql = MySQL(app)


@app.route('/')
def index():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM articles")
    articles = cur.fetchall()
    cur.close()
    return render_template('index.html', articles=articles)


@app.route('/article/<int:article_id>')
def article(article_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM articles WHERE article_id = %s", (article_id,))
    articles = cur.fetchone()
    cur.close()
    # Render the article.html template and pass the article to the template context
    return render_template('article.html', article=articles)


@app.route('/articlsummary/<int:article_id>')
def articlesummary(article_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM articles WHERE article_id = %s", (article_id,))
    articles = cur.fetchone()
    cur.close()
    articles = list(articles)  # Convert tuple to a list
    articles[2] = nlp.fun(articles[2])  # Modify the desired element
    articles = tuple(articles)  # Convert back to a tuple if needed

    # Render the article.html template and pass the article to the template context
    return render_template('articleSummary.html', article=articles)


if __name__ == '__main__':
    app.run()

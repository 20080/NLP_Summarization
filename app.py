import re

from flask import Flask, render_template
import pandas as pd
from flask_mysqldb import MySQL
from markupsafe import Markup

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


@app.route('/articlesummary/<int:article_id>')
def articlesummary(article_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM articles WHERE article_id = %s", (article_id,))
    article1 = cur.fetchone()
    cur.execute("SELECT tag, article_id FROM tags WHERE article_id = %s", (article_id,))
    tags = cur.fetchall()
    cur.close()

    summary = nlp.fun(article1[2])  # Generate the summary
    print(tags)
    # Iterate over the tags and replace them with highlighted and linked versions
    for tag in tags:
        tag_name = tag[0]
        highlighted_tag = f'<a href="/relatedarticles/{tag_name}" style="color: blue;">{tag_name}</a>'
        summary = re.sub(fr'\b{re.escape(tag_name)}\b', highlighted_tag, summary, count=1)

    article1 = list(article1)  # Convert tuple to a list
    article1[2] = summary  # Update the summary in the article list
    article1 = tuple(article1)  # Convert back to a tuple if needed

    # Render the articleSummary.html template and pass the article to the template context
    return render_template('articleSummary.html', article=article1, summary=Markup(article1[2]))


@app.route('/relatedarticles/<string:tag_name>')
def relatedarticles(tag_name):
    cur = mysql.connection.cursor()
    cur.execute("SELECT article_id FROM tags WHERE tag = %s", (tag_name,))
    article_ids = cur.fetchall()
    articles = []
    for article_id in article_ids:
        cur.execute("SELECT * FROM articles WHERE article_id = %s", (article_id[0],))
        article = cur.fetchone()
        articles.append(article)

    cur.close()

    return render_template('relatedarticles.html', tag_name=tag_name, articles=articles)


if __name__ == '__main__':
    app.run()

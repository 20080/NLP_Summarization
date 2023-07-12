from flask import Flask, render_template, request

import nlp

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']

        # Check if the input text has at least 50 lines
        lines = text.split('\n')
        if len(lines) < 10:
            return render_template('index.html', error='Input text must have at least 10 lines')

        # Get the selected models from the checkboxes
        selected_models = request.form.getlist('models')

        # Perform text summarization using the selected models
        summaries = {}
        for model in selected_models:
            line = int(request.form['lines'])
            if model == 'google_news_vector':
                summary = nlp.fungoogle(text, line)
            elif model == 'glove':
                summary = nlp.funglove(text, line)
            elif model == 'bert':
                summary = nlp.funbert(text, line)
            summaries[model] = summary

        # Render the summaries on the index page
        return render_template('index.html', summaries=summaries, selected_models=selected_models, line=line, text=text)

    return render_template('index.html', line=7)


if __name__ == '__main__':
    app.run()

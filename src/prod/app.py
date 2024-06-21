from workerA import get_accuracy, get_predictions

from flask import (
   Flask,
   request,
   render_template
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template(
            "home.html"
            )


@app.route("/accuracy", methods=['POST', 'GET'])
def accuracy():
    if request.method == 'POST':
        r = get_accuracy.delay()
        a = r.get()
        return '<h1>The accuracy is {}</h1>'.format(a)

    return '''<form method="POST">
    <input type="submit">
    </form>'''


@app.route("/predict", methods=['POST', 'GET'])
def predictions():
    results = get_predictions.delay()
    predictions = results.get()

    results = get_accuracy.delay()
    mae, r2 = results.get()
    final_results = predictions

    return render_template(
            'result.html',
            mae=mae,
            final_results=final_results,
            r2=r2
            )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)

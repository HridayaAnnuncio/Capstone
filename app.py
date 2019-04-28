import flask
import pickle
import pandas as pd
app = flask.Flask(__name__, template_folder='templates',static_folder='static')
# Use pickle to load in the pre-trained model.
with open('model/storm_rf_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
deg_sym = '°'
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        latitude = flask.request.form['latitude']
        longitude = flask.request.form['longitude']
        sustained_winds = flask.request.form['sustainedwinds']
        input_variables = pd.DataFrame([[latitude,longitude,sustained_winds]],
                                       columns=['latitude','longitude','sustained_winds'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Latitude':latitude+deg_sym,
                                                     'Longitude':longitude+deg_sym,
                                                     'Sustained Winds':sustained_winds+ " Knots"},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

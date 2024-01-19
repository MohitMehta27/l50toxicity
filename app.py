import flask
from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        
        data = CustomData (
            CIC0=request.form.get('CIC0'),
            SM1_Dz_Z=request.form.get('SM1_Dz_Z'),
            GATS1i=request.form.get('GATS1i'),
            NdsCH=request.form.get('NdsCH'),
            NdssC=request.form.get('NdssC'),
            MLOGP=request.form.get('MLOGP')
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0], 2)

        return render_template('results.html', final_result=results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

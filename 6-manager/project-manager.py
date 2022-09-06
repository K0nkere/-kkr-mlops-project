import prefect
import json

from datetime import datetime
from dateutil.relativedelta import relativedelta

import prefect_monitoring_report
import prefect_model

from flask import Flask, request, jsonify
app = Flask("Model-manager")

@app.route("/manager", methods=['POST'])
def project_manager():

    signal = request.get_json()

    print(f"Getting {signal['finished']} on {signal['current_date']}")
    
    report = prefect_monitoring_report.batch_analyze(
                current_date = signal['current_date'],
                target_file = '../target.csv'
                )[0]
    
    retrain = report['data_drift']['data']['metrics']['prediction']['drift_detected']
    print(f"Drift status on {signal['current_date']} prediction:", retrain)

    if retrain:
        
        retrain_date = datetime.strptime(signal['current_date'], "%Y-%m-%d")

        if retrain_date < datetime(2015, 8, 1):
            retrain_date = datetime.strftime(retrain_date, "%Y-%m-%d")
            print("... Running retraining of the model ...")
            prefect_model.main(current_date = retrain_date, periods = 5)
            
        else:
            retrain_date = datetime.strftime(retrain_date, "%Y-%m")
            print(f"... Unable to retrain, waiting for the data over {retrain_date}...")
        
    return jsonify("OK")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9898)

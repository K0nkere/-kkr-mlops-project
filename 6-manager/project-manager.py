import prefect
import json

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

    print(f"Drift status on {signal['current_date']} prediction:", report['data_drift']['data']['metrics']['prediction']['drift_detected'])

    return jsonify("OK")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9898)

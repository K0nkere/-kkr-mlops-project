# pylint: disable=redefined-builtin
"""
Manager of services
"""

from flask import Flask, globals, jsonify, request

globals.create_report = True
globals.current_date = "2015-5-30"
globals.retrain = True

app = Flask("Model-manager")


@app.route("/manager", methods=['POST'])
def project_manager():
    """
    Contains logic of service management
    """

    signal = request.get_json()

    if "create_report" in signal.keys():
        answer = {
            "current_date": globals.current_date,
            "target_file": "targets/target.csv",
            "create_report": globals.create_report,
        }

        globals.create_report = False
        print(
            f"... Setting create_report varible to {globals.create_report} ..."
        )
        return jsonify(answer)

    if signal.get("service") == "report":
        print("... Getting report ... ")

        if signal["drift"]:
            globals.current_date = signal["retrain_period"]

            print(
                f'... Drift status: {signal["drift"]} on {globals.current_date} ...'
            )
            print(
                f"... Need to retrain model on the data up to {globals.current_date} ..."
            )

            globals.retrain = signal["drift"]
            print(f"... Setting retrain varible to {globals.retrain} ...")

        return jsonify("OK")

    if signal.get("service") == "sending_stream":
        print("... Getting signal from sending stream ... ")

        if signal["finished"]:
            globals.current_date = signal["current_date"]

            globals.create_report = True
            print(
                f"... Setting create_report varible to {globals.create_report} ..."
            )

        return jsonify("OK")

    if signal.get("service") == "training_model":
        print("... Getting request from training service ... ")

        if globals.retrain:
            answer = {"current_date": globals.current_date}

            print(
                f"... Launching retraining for the data up tp {globals.current_date} ..."
            )

            globals.retrain = False
            print(f"... Setting retrain varible to {globals.retrain} ...")

            return jsonify(answer)

        else:
            answer = {"current_date": False}
            print(
                "... No need to retrain model. Waiting for the latest data ..."
            )
            return jsonify(answer)

    # print(f"Getting signqal from {signal['service']} service on {signal['current_date']}")

    # report = prefect_monitoring_report.batch_analyze(
    #             current_date = signal['current_date'],
    #             target_file = '../targets/target.csv'
    #             )[0]

    # retrain = report['data_drift']['data']['metrics']['prediction']['drift_detected']
    # print(f"Drift status on {signal['current_date']} prediction:", retrain)

    # if retrain:

    #     retrain_date = datetime.strptime(signal['current_date'], "%Y-%m-%d")

    #     if retrain_date < datetime(2015, 8, 1):
    #         retrain_date = datetime.strftime(retrain_date, "%Y-%m-%d")
    #         print("... Running retraining of the model ...")
    #         prefect_model.main(current_date = retrain_date, periods = 5)

    #     else:
    #         retrain_date = datetime.strftime(retrain_date, "%Y-%m")
    #         print(f"... Unable to retrain, waiting for the data over {retrain_date}...")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9898)

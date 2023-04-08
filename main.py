from flask import Flask, request, Response, jsonify
from werkzeug.utils import secure_filename
import json, os
import tshirt_size

app = Flask(__name__)


@app.route('/get_size', methods=['POST', 'GET'])
def getSize():
    try:
        f = request.files['file']
        filename = secure_filename(f.filename)
        print("Received filename {0}".format(filename))
        f.save(os.path.join(os.getcwd(), filename))
        ans = tshirt_size.get_size(filename)
        return jsonify(Tshirtsize=ans)
    except Exception as e:
        return jsonify(Tshirtsize='Error')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=1777)
    # app.run()

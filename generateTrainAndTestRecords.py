import os
import json

with open('./../config.json') as json_file:
    config = json.load(json_file)

runName = config["runName"]
runDir = './../data/' + runName + '/'

trainingCall = "python ./generate_tfrecord.py --csv_input=data/trainLabels.csv --output_path=./../data/" + runName + "/train.record --image_dir=./../data/" + runName + "/images"

testingCall = "python ./generate_tfrecord.py --csv_input=data/testLabels.csv --output_path=./../data/" + runName + "/test.record --image_dir=./../data/" + runName + "/images"

os.system(trainingCall)
os.system(testingCall)

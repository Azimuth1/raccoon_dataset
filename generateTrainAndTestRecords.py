import os
import json
import argparse
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        help="name of the run  which is used as path to the data directory",
    )
    args = parser.parse_args()

    with open("./../config.json") as json_file:
        config = json.load(json_file)

    config_original = copy.deepcopy(config)

    config["runName"] = args.run_name

    with open("./../config.json", "w") as json_file:
        json.dump(config, json_file)
    
    runName = config["runName"]
    runDir = './../data/' + runName + '/'

    trainingCall = "python ./generate_tfrecord.py --csv_input=./../data/" + runName + "/labels_train.csv --output_path=./../data/" + runName + "/train.record --image_dir=./../data/" + runName + "/images"
    testingCall = "python ./generate_tfrecord.py --csv_input=./../data/" + runName + "/labels_test.csv --output_path=./../data/" + runName + "/test.record --image_dir=./../data/" + runName + "/images"

    os.system(trainingCall)
    os.system(testingCall)

    with open("./../config.json", "w") as json_file:
        json.dump(config_original, json_file)

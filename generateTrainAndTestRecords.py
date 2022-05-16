import os
import json
import argparse
import copy
import pathlib

root_path = str(pathlib.Path(__file__).parent.resolve()) + "/../../../"
configs_path = root_path + "src/configs/"
main_data_path = root_path + "src/data/"
with open(configs_path + "config.json") as json_file:
    config = json.load(json_file)

runName = config["runName"]
runDir = main_data_path + runName + "/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        help="name of the run  which is used as path to the data directory",
    )
    args = parser.parse_args()


    with open(configs_path + "config.json") as json_file:
        config = json.load(json_file)

    config["runName"] = args.run_name

    with open(configs_path + "config.json", "w") as json_file:
        json.dump(config, json_file)
    
    runName = config["runName"]
    runDir = main_data_path + runName + '/'

    trainingCall = "python ./generate_tfrecord.py --csv_input=" + main_data_path + runName + "/labels_train.csv --output_path=" + main_data_path + runName + "/train.record --image_dir=" + main_data_path + runName + "/images"
    testingCall = "python ./generate_tfrecord.py --csv_input=" + main_data_path + runName + "/labels_test.csv --output_path=" + main_data_path + runName + "/test.record --image_dir=" + main_data_path + runName + "/images"

    os.system(trainingCall)
    os.system(testingCall)

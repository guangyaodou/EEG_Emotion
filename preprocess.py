import os
import argparse
import pandas as pd


def preprocess(subject_id):
    current_path = os.getcwd()
    event_marker_dic = '/Eventmarker_Behavior_Data/'
    behavior_data_path = os.path.join(current_path + event_marker_dic,
                                      f"subject_{subject_id}/{subject_id}_pilot_study.csv")

    subject_behavior = pd.read_csv(behavior_data_path)

    # remove first two lines
    subject_behavior = subject_behavior.iloc[2:].reset_index(drop=True)

    # select the first 120 lines of behavior data
    df = subject_behavior.iloc[:120]

    new_column = []
    for i in df["mouse.clicked_name"]:
        scale_level = int(i[-3])
        if 1 <= scale_level and scale_level <= 3:
            new_column.append("negative")
        elif 4 <= scale_level and scale_level <= 6:
            new_column.append("neutral")
        else:
            new_column.append("positive")
    assert len(new_column) == len(df["mouse.clicked_name"])

    # Read original event marker
    OriginalEventMarker_data_path = os.path.join(current_path + event_marker_dic,
                                                 f"subject_{subject_id}/{subject_id}_OriginalEventMarker.txt")
    event_marker = pd.read_csv(OriginalEventMarker_data_path, delim_whitespace=True)

    event_marker = event_marker[(event_marker["type"] == "p") |
                                (event_marker["type"] == "u") |
                                (event_marker["type"] == "g")].iloc[:120]

    event_marker["type"] = new_column

    # specify path for export
    export_path = os.path.join(current_path + event_marker_dic, f"subject_{subject_id}/{subject_id}_NewEventMarker.txt")
    # export DataFrame to text file
    with open(export_path, 'w') as f:
        df_string = event_marker.to_string(index=False)
        f.write(df_string)

    return event_marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='EEG Emotion event marker preprocessing',
        description='replaces the original event marker with subjects behavior valence level, and generates a new event marker file')

    parser.add_argument("--subject_id", help="the id of the subject", required=True)
    args = parser.parse_args()

    subject_id = args.subject_id
    preprocess(subject_id)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# MODIFY if necessary
subject_lst = ["02", "21"]

number_of_folds = 10

# If true, we ignore neural case and only calssfiy between positive and negative
binary_classification = True

number_of_neighbors = 5

# FOR POWER DATA:
# Frequency bands for power data
power_band = ["alpha", "beta", "delta", "theta"]
# the channel you want to add for the power data
channel = ["AF7", 'AF8']

NORMALIZE_METHODS = "standard_scalar",  # normalizer, "standard_scalar", "MinMax", or None

# DO NOT MODIFY
lower_bound = 10
# Because we are cutting the length of each unit to match the length of power
upper_bound = 550
power_length = 153

NUM_OF_POWER_COLUMNS = 5

# Read negative, neutral, and positive data from data folder
states = ["negative", "neutral", "positive"]
bands = ["alpha", "beta", "delta", "theta"]


def assign_units_to_folds(df, folds, lower_bound, upper_bound):
    """
    Execute before k-fold cross-validation: extract each unit, assign it to different folds in order
    df: The dataframe that contains data of each patient
    lower_bound: Experiment time's lower bound indicating the beginning of each unit
    upper_bound: Experiment time's upper bound indicating the ending of each unit
    """
    num_folds = len(folds)
    num_rows = len(df)
    j = 1
    i = fold_pointer = 0
    while j <= num_rows - 1:
        prev_time = df.iloc[j - 1]["Time"]
        time = df.iloc[j]["Time"]
        # if the time jumps from upper_bound(250) to a time smaller than the lower_bound (-80)
        # we find a unit
        if (time < lower_bound and prev_time > upper_bound):
            unit = df.iloc[i:j]
            folds[fold_pointer].append(unit)
            fold_pointer = (fold_pointer + 1) % num_folds
            i = j
        j = j + 1
    last_unit = df.iloc[i: j]
    folds[fold_pointer].append(last_unit)


def concat_dataframes(fold_list, remove_columns_names):
    """
    concatenate lists of dataframes to one dataframe and drop the specified columns if needed
    fold_list: a list of folds
    remove_columns_names: a list of names of columns you want to exclude
    """
    folds_concat = []
    for fold in fold_list:
        folds_concat.append(pd.concat(fold, ignore_index=True).drop(columns=remove_columns_names))
    return folds_concat


def get_features(subject_data):
    column_names = subject_data.columns
    alpha_columns = [i for i in column_names if "alpha" in i]
    beta_columns = [i for i in column_names if "beta" in i]
    theta_columns = [i for i in column_names if "theta" in i]

    alpha = subject_data.loc[:, alpha_columns]
    beta = subject_data.loc[:, beta_columns]
    theta = subject_data.loc[:, theta_columns]

    alpha_std = np.std(alpha, axis=1)
    beta_std = np.std(beta, axis=1)
    theta_std = np.std(theta, axis=1)

    alpha_mean = np.mean(alpha, axis=1)
    beta_mean = np.mean(beta, axis=1)
    theta_mean = np.mean(theta, axis=1)

    # Concate feature
    feature = np.array([theta_std, theta_mean, alpha_std, alpha_mean, beta_std, beta_mean])
    feature = feature.T

    return feature


def match_power(df, df_power, index):
    """
    df: The frequency dataframe
    df_power: The power dataframe
    index: the starting index of power name
    """
    num_columns = len(df_power.columns)
    assert num_columns == NUM_OF_POWER_COLUMNS
    res = pd.DataFrame()
    zero_idx = df.index[df['Time'] == 0].tolist()
    for idx in zero_idx:
        df_temp = df.iloc[idx:idx + power_length + 1, :]
        res = pd.concat([res, df_temp], axis=0)
    res = res.reset_index()
    assert len(res) == len(df_power)
    column_names = [f"Power{i}" for i in range(index, index + num_columns)]
    df_power = df_power.set_axis(column_names, axis=1)

    df_res = pd.concat([res, df_power], axis=1)
    return df_res


def f_importances(coef, names, subject_id):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.rcParams.update({'font.size': 5})
    plt.tick_params(axis='y', labelsize=5)
    plt.savefig(f'output/{subject_id}_LSVM_FeatureImportance.png', dpi=2000, bbox_inches="tight")
    plt.show()


for subject_id in subject_lst:
    neg_neu_pos = []

    subject_id = subject_id

    for state in states:
        df = pd.DataFrame()
        for band in bands:
            df_temp = pd.read_csv(f"data_newcut/{subject_id}_processed_{state}_flt_{band}.csv")
            df_temp = df_temp.rename(columns={"TP9": f"TP9_{band}",
                                              "AF7": f"AF7_{band}",
                                              "AF8": f"AF8_{band}",
                                              "TP10": f"TP10_{band}"})
            # remove the time column for beta and theta
            if band in ["beta", "theta", "delta"]:
                df_temp = df_temp.drop([f"Time"], axis=1)

            df = pd.concat([df, df_temp], axis=1)

        # Remove times that are smaller than 0
        df = df[df['Time'] >= 0].reset_index(drop=True)
        neg_neu_pos.append(df)

    # states are the same as the previous ones: "negative", "neutral", "positive"
    for i, state in enumerate(states):
        s_state = neg_neu_pos[i]
        power_idx = 1
        for band in power_band:
            for cn in channel:
                tmp_power = pd.read_csv(
                    f"data_newcut/Power/{subject_id}_processed_{state}_flt_{band}_{cn}_power.csv",
                    header=None)
                tmp_power = tmp_power.transpose()

                s_state = match_power(s_state, tmp_power, power_idx)
                lst_column = [i for i in range(power_idx, power_idx + NUM_OF_POWER_COLUMNS)]
                s_state = s_state.drop(columns=["index"])

                # Add average of the previous five power columns
                s_state[f"avg_{band}_power_{cn}"] = s_state.iloc[:, -NUM_OF_POWER_COLUMNS:].sum(
                    axis=1) / NUM_OF_POWER_COLUMNS
                power_idx = power_idx + NUM_OF_POWER_COLUMNS

        neg_neu_pos[i] = s_state

    subject_negative = neg_neu_pos[0]
    subject_neutral = neg_neu_pos[1]
    subject_positive = neg_neu_pos[2]

    # Suppose negative = 0; positive = 1; and neutral = 2;
    subject_negative["y"] = 0
    subject_neutral["y"] = 2
    subject_positive["y"] = 1

    # Concatenate all three datasets
    subject_data = pd.concat([subject_negative, subject_neutral, subject_positive], ignore_index=True)
    subject_data = subject_data.reset_index(drop=True)

    # added mean and standard deviation of each bands
    feature = get_features(subject_data)
    df_newFeature = pd.DataFrame(feature, columns=['theta_std', 'theta_mean', 'alpha_std',
                                                   'alpha_mean', 'beta_std', 'beta_mean'
                                                   ])
    df = pd.concat([subject_data, df_newFeature], axis=1)

    # re-normalize the data (for non-y columns)
    y_column = df.pop("y")
    column_names = df.columns

    cols = df.columns[df.columns != 'Time']
    if NORMALIZE_METHODS == "normalizer":
        df[cols] = sklearn.preprocessing.normalize(df[cols], axis=0)
    elif NORMALIZE_METHODS == "standard_scalar":
        df[cols] = sklearn.preprocessing.scale(df[cols]),
    elif NORMALIZE_METHODS == "MinMax":
        df[cols] = sklearn.preprocessing.MinMaxScaler().fit_transform(df[cols])

    df = pd.DataFrame(df, columns=column_names)
    df["y"] = y_column.replace(np.nan, 0)

    if binary_classification:
        df = df[df['y'] != 2]

    folds = [[] for i in range(number_of_folds)]
    assign_units_to_folds(df, folds, lower_bound, upper_bound)
    columns = df.columns

    columns_to_remove = []
    columns_to_remove.append(columns[0])
    columns_to_remove.extend([i for i in columns if "TP" in i])
    # columns_to_remove.extend(columns[0:13])
    print("Columns removed", columns_to_remove)
    folds_concat = concat_dataframes(folds, columns_to_remove)

    # SVMs take long time to run!
    names = [
        #     'XGBoost',
        #     "Adaboost",
        #     "RandomForest",
        #     "GradientBoost",
        'Nearest Neighbors',
        #     'LDA',
        #     'RBF SVM',
        #     "Linear SVM",
    ]

    acc_res = {}
    for name in names:
        accuracy_lst = []
        for _ in range(number_of_folds):
            if name == "Adaboost":
                clf = AdaBoostClassifier()
            if name == "RandomForest":
                clf = RandomForestClassifier(n_estimators=100, max_features="sqrt", oob_score=True)
            if name == "GradientBoost":
                clf = GradientBoostingClassifier()
            if name == 'Nearest Neighbors':
                clf = KNeighborsClassifier(n_neighbors=number_of_neighbors)
            if name == "LDA":
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            if name == "RBF SVM":
                clf = SVC(gamma=2, C=1)
            if name == "Linear SVM":
                clf = SVC(kernel="linear", C=0.025)
            if name == "XGBoost":
                clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric='mlogloss', random_state=42,
                                        use_label_encoder=False)
            train_data = pd.concat(folds_concat[:-1], ignore_index=True)
            # take the last fold as the test set
            test_data = folds_concat[-1]
            # move the last fold to the beginning of the list of folds
            folds_concat = folds_concat[-1:] + folds_concat[:-1]
            train_X = train_data.iloc[:, :-1]
            train_Y = train_data.iloc[:, -1]
            test_X = test_data.iloc[:, :-1]
            test_Y = test_data.iloc[:, -1]
            clf.fit(train_X, train_Y)
            y_predict = clf.predict(test_X)
            accuracy = metrics.accuracy_score(y_predict, test_Y)
            accuracy_lst.append(accuracy)
        if name == "Linear SVM":
            f_importances(clf.coef_[0], clf.feature_names_in_, subject_id)
        avg_acc = round(sum(accuracy_lst) / len(accuracy_lst), 3)
        acc_res[name] = avg_acc
        print(f"{name} yields accuracy of {avg_acc}")

    print(subject_id, acc_res)

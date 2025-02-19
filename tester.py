import importlib, sys, json, time
from datetime import datetime
import pandas as pd
import numpy as np
import os
from main import UserPredictor

lower = 50
upper = 75
max_sec = 60
module_name = "main"
test_set = "data/test1"

def main():
    t0 = time.time()
    print("Fitting+Predicting...")

    # step 1: fit
    model = UserPredictor()
    train_users = pd.read_csv(os.path.join("data", "train_users.csv"))
    train_logs = pd.read_csv(os.path.join("data", "train_logs.csv"))
    train_y = pd.read_csv(os.path.join("data", "train_y.csv"))
    model.fit(train_users, train_logs, train_y)

    # step 2: predict
    test_users = pd.read_csv(f"{test_set}_users.csv")
    test_logs = pd.read_csv(f"{test_set}_logs.csv")
    y_pred = model.predict(test_users, test_logs)

    # step 3: grading based on accuracy
    y = pd.read_csv(f"{test_set}_y.csv")
    accuracy = (y["y"] == y_pred).sum() / len(y) * 100
    grade = round((np.clip(accuracy, lower, upper) - lower) / (upper - lower) * 100, 1)

    t1 = time.time()
    sec = t1-t0
    assert sec < max_sec
    warn_sec = 0.75 * max_sec
    if sec > warn_sec:
        print("="*40)
        print("WARNING!  Tests took", sec, "seconds")
        print("Maximum is ", max_sec, "seconds")
        print(f"We recommend keeping runtime under {warn_sec} seconds to be safe.")
        print("Variability may cause it to run slower for us than you.")
        print("="*40)

    # output results
    results = {"score":grade,
               "accuracy": accuracy,
               "date":datetime.now().strftime("%m/%d/%Y"),
               "latency": sec}
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Result:\n" + json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler 

pd.set_option('future.no_silent_downcasting', True)

class UserPredictor:
    def __init__(self):        
        self.model = Pipeline([ 
            ("pf", PolynomialFeatures(degree = 2, include_bias = False)),
            ("std", StandardScaler()),
            ("lr", LogisticRegression(fit_intercept = False, max_iter = 1000)),
        ])  
    
    def add_logs_as_features(self, users, logs):
        total_seconds = logs.groupby("user_id")["seconds"].sum().reset_index()
        total_visits = logs.groupby("user_id")["url"].count().reset_index()
                
        total_seconds.rename(columns = {"seconds": "total_seconds"}, inplace = True)
        total_visits.rename(columns = {"url": "total_visits"}, inplace = True)
        
        users = users.merge(total_seconds, on = "user_id", how = "left")
        users = users.merge(total_visits, on = "user_id", how = "left")

        users["badge"] = users["badge"].replace({"gold": 3, "silver": 2, "bronze": 1}).fillna(0).astype(int)
        
        users.fillna({"total_seconds": 0, "total_visits": 0}, inplace = True)
        
        return users

    def fit(self, users, logs, y):
        users = self.add_logs_as_features(users, logs)
        df = pd.merge(users, y, on = "user_id", how = "inner")        
        xcols = ["past_purchase_amt", "age", "total_seconds", "total_visits", "badge"]
        ycol = "y"
        self.model.fit(df[xcols], df[ycol])

    def predict(self, users, logs):
        users = self.add_logs_as_features(users, logs)        
        xcols = ["past_purchase_amt", "age", "total_seconds", "total_visits", "badge"]
        
        return self.model.predict(users[xcols])
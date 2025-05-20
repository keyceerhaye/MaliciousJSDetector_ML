import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def parse_logs(self, log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
        return logs

    def extract_features(self, logs):
        features = {}
        # Count features
        features['eval_count'] = len(logs['js_api_calls'])
        features['alert_count'] = len(logs['alerts'])
        features['dom_change_count'] = len(logs['dom_changes'])
        features['network_request_count'] = len(logs['network_requests'])
        features['cpu_usage_count'] = len(logs['cpu_usage'])
        # Binary flags
        features['has_eval'] = 1 if features['eval_count'] > 0 else 0
        features['has_alert'] = 1 if features['alert_count'] > 0 else 0
        features['has_dom_change'] = 1 if features['dom_change_count'] > 0 else 0
        features['has_network_request'] = 1 if features['network_request_count'] > 0 else 0
        features['has_cpu_usage'] = 1 if features['cpu_usage_count'] > 0 else 0
        return features

    def create_feature_vector(self, log_files):
        feature_list = []
        for log_file in log_files:
            logs = self.parse_logs(log_file)
            features = self.extract_features(logs)
            feature_list.append(features)
        df = pd.DataFrame(feature_list)
        # Normalize numerical features
        numerical_features = ['eval_count', 'alert_count', 'dom_change_count', 'network_request_count', 'cpu_usage_count']
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        return df

    def save_features(self, df, output_path):
        df.to_csv(output_path, index=False)

if __name__ == '__main__':
    extractor = FeatureExtractor()
    log_files = ['behavior_logs.json']  # List of log files
    feature_df = extractor.create_feature_vector(log_files)
    extractor.save_features(feature_df, 'features.csv') 
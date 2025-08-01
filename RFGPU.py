from river import drift
from cuml.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time

class RFGPU:
    def __init__(self, stream, burnout_window_size, exp_length, batch_size, ensemble_size=100, correctly_classified= 0.3, detector_type=drift.binary.DDM(drift_threshold=4)):
        if isinstance(stream, pd.DataFrame):
            print("É Dataframe!")
            stream = self._encode_onehot(stream, target_column=stream.columns[-1])
            self.stream = self.dataframe_to_stream(stream)
        else:
            print("É Stream!")
            self.stream = stream

        self.burnout_window_size = burnout_window_size
        self.exp_length = exp_length
        self.batch_size = batch_size
        self.buffer = []
        self.n_observed = 0
        self.correctly_classified = correctly_classified
        self.ensemble_size = ensemble_size

        if isinstance(detector_type, type):
            self.detector = detector_type()
        else:
            self.detector = detector_type

        burnout_window = self._get_initial_instances()
        labels_bw = [y for (_, y) in burnout_window]
        while len(set(labels_bw)) < 2:
            x_extra, y_extra = next(self.stream)
            burnout_window.append((list(x_extra.values()), y_extra))
            labels_bw = [y for (_, y) in burnout_window]
        self.model = self._init_rf(burnout_window)

    def _encode_onehot(self, df, target_column):
        print("Etapa de Encoding\n")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        if not categorical_cols:
            return df.copy()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(df[categorical_cols])
        encoded_col_names = encoder.get_feature_names_out(categorical_cols)
        df_encoded = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df.index)
        df_numeric = df.drop(columns=categorical_cols)
        df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
        df_result = pd.concat([df_numeric, df_encoded], axis=1)
        return df_result.fillna(0)

    def dataframe_to_stream(self, df):
        target_column = df.columns[-1]
        df = df.copy()
        if df[target_column].dtype == object or str(df[target_column].dtype).startswith('category'):
            df[target_column] = df[target_column].astype("category").cat.codes
        for _, row in df.iterrows():
            x = row.drop(target_column).to_dict()
            y = row[target_column]
            yield x, y

    def _get_initial_instances(self):
        print("Juntando instancias para a Burnout window\n")
        burnout_window = []
        for _ in range(self.burnout_window_size):
            x, y = next(self.stream)
            burnout_window.append((list(x.values()), y))
        return burnout_window

    def _init_rf(self, data):
        print("Treinamento inicial\n")
        rf = RandomForestClassifier(n_estimators=self.ensemble_size, max_depth=10)
        X, y = zip(*data)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        rf.fit(X, y)
        return rf

    def run(self):
        inicio = time.time()
        true_labels = []
        predicted_labels = []
        correct_buffer = []
        stream_exhausted = False

        while self.n_observed < self.exp_length and not stream_exhausted:
            batch = []
            y_batch = []

            for _ in range(self.batch_size):
                if self.n_observed >= self.exp_length:
                    break
                try:
                    x, y = next(self.stream)
                    batch.append(list(x.values()))
                    y_batch.append(y)
                    self.n_observed += 1
                except StopIteration:
                    stream_exhausted = True
                    break

            X_batch = np.array(batch, dtype=np.float32)
            y_pred_batch = self.model.predict(X_batch)

            for x, y, y_pred in zip(batch, y_batch, y_pred_batch):
                error = float(y != y_pred)
                self.detector.update(error)

                if error == 1:
                    self.buffer.append((x, y))
                else:
                    if np.random.rand() < self.correctly_classified:
                        correct_buffer.append((x, y))

                if self.detector.drift_detected:
                    print(f"Drift detectado em {self.n_observed}!")
                    combined_buffer = self.buffer + correct_buffer

                    if len(combined_buffer) > 100:
                        labels = [y for (_, y) in combined_buffer]
                        if len(set(labels)) > 1:
                            self.model = self._init_rf(combined_buffer)
                            self.buffer = []
                            correct_buffer = []
                        else:
                            print(f"Buffer tem {len(combined_buffer)} instâncias mas só uma classe presente. Aguardando mais dados para re-treinamento.")
                    else:
                        print(f"Buffer insuficiente para re-treinamento ({len(combined_buffer)} instâncias). Aguardando mais dados.")

                true_labels.append(y)
                predicted_labels.append(y_pred)

        fim = time.time()
        print(f"\nTempo total de execução: {fim - inicio:.2f} segundos")
        print("\nRelatório de Classificação:")
        report = classification_report(true_labels, predicted_labels, zero_division=0, output_dict=True)
        print(report)
        return report

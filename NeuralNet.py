#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)




    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input

        # Define column names based on adult.names
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]

        # Load the data
        df = self.raw_input
        df.columns = columns

        # Replace '?' with NaN
        df.replace('?', np.nan, inplace=True)

        # Drop fnlwgt as it's not typically used as a feature
        df.drop('fnlwgt', axis=1, inplace=True)

        # Separate features and target
        X = df.drop('income', axis=1)
        y = df['income']

        # Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(y)  # 0 for <=50K, 1 for >50K

        # Identify categorical and numerical columns
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']
        numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)

        # If the transformer returns a sparse matrix (common with OneHotEncoder), convert it to dense
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()

        # Create DataFrame for processed features
        X_df = pd.DataFrame(X_processed, columns=feature_names)

        # Combine features and target into a single DataFrame
        self.processed_data = X_df.assign(income=y)

        return 0


    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        histories = []
        results = []

        def build_model(input_dim, num_hidden, activation, lr):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Input(shape=(input_dim,)))
            for _ in range(num_hidden):
                model.add(tf.keras.layers.Dense(64, activation=activation))
            model.add(tf.keras.layers.Dense(1, activation=activation))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model

        for activation in activations:
            for lr in learning_rate:
                for epochs in max_iterations:
                    for n_hidden in num_hidden_layers:
                        model = build_model(X_train.shape[1], n_hidden, activation, lr)
                        history = model.fit(
                            X_train,
                            y_train,
                            validation_split=0.2,
                            epochs=epochs,
                        )
                        test_loss, test_acc = model.evaluate(X_test, y_test)

                        train_loss = history.history['loss'][-1]
                        train_acc = history.history['accuracy'][-1]
                        val_loss = history.history['val_loss'][-1]
                        val_acc = history.history['val_accuracy'][-1]

                        results.append({
                            'activation': activation,
                            'lr': lr,
                            'epochs': epochs,
                            'hidden_layers': n_hidden,
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'test_loss': test_loss,
                            'test_acc': test_acc,
                        })

                        histories.append((f"{activation}-lr{lr}-e{epochs}-h{n_hidden}", history))

                        print(
                            f"{activation} lr={lr} epochs={epochs} hidden={n_hidden} -> "
                            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
                        )

        print(histories)
        
        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.
        
        plt.figure(figsize=(12, 36))
        for i, act in enumerate(activations, start=1):
            ax = plt.subplot(3, 1, i)
            for name, hist in histories:
                if name.startswith(act + '-'):
                    ax.plot(hist.history['accuracy'], label=f"{name}-train")
                    # ax.plot(hist.history['val_accuracy'], linestyle='--', label=f"{name}-val")
            ax.set_title(f"Training accuracy ({act})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize='x-small', ncol=2, loc='upper left')
        plt.tight_layout()
        plt.show()
        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("adult.data") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()

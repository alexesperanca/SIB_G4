import numpy as np
from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor


class SupervisedModelOptimization:
    def __init__(
        self,
        train_input: np.ndarray,
        train_output: np.ndarray,
        test_input: np.ndarray = None,
        test_output: np.ndarray = None,
    ):
        """SupervisedModelOptimization is a class to test multiple Supervised Machine Learning models, while tuning the respective best hyperparameters.
        If no test data is assigned, we split 30% of the inputted train data to compose the testing data.

        Args:
            train_input (np.ndarray): Train data input.
            train_output (np.ndarray): Train data output.
            test_input (np.ndarray, optional): Test data input. Defaults to None.
            test_output (np.ndarray, optional): Test data output. Defaults to None.
        """
        if not test_input or not test_output:
            print("No assigned test data, splitting the inputted train data")
            (
                self.train_input,
                self.train_output,
                self.test_input,
                self.test_output,
            ) = self._split_data(train_input, train_output)
        else:
            self.train_input = train_input
            self.train_output = train_output
            self.test_input = test_input
            self.test_output = test_output

        # Attributes
        self.model_estimator = None
        self.model = None
        self.predictions = None

    def summary(self):
        """Method to print the important information. Display of the dataset dimensions and model prediction + results."""
        # Check Dimensions
        print(
            "Input and output training Dimensions:",
            self.train_input.shape,
            self.train_output.shape,
        )
        print(
            "Input and output testing Dimensions:",
            self.test_input.shape,
            self.test_output.shape,
        )

        if self.predictions is not None:
            print()
            print("=" * 100)
            print("Model:", self.model_estimator.__class__.__name__)

            print("\n**First 10 Values Previsions**")
            aligned_predictions = list(zip(self.test_output, self.predictions))[:10]
            print("Testing values | Model predictions")
            print(
                "\n".join(
                    f"{round(values[0], 1)} | {values[1]}"
                    for values in aligned_predictions
                )
            )

            best_params = None
            best_score = self.score()
            
            # If we have hyperparameter tuning
            if self.model.__class__.__name__ != self.model_estimator.__class__.__name__:
                best_params = self.model.best_params_

            print("\n**Results**")
            print("Best parameters set found on development set: \n", best_params,)
            print("Best/lowest score (mean_squared_error): \n", best_score)
            print("=" * 100)

    def _split_data(
        self, input: np.ndarray, output: np.ndarray, division_rate: int = 0.3
    ):
        """Method to split the data according to the input and output samples. The division rate may be altered and is predefined as 30% will be considered as testing data.

        Args:
            input (np.ndarray): Input samples.
            output (np.ndarray): Input data.
            division_rate (int, optional): Division rate to obtain the test data. Defaults to 0.3.

        Returns:
            tuple[np.ndarray]: Tuple with each of the divided data.
        """
        num_test = int(input.shape[0] * division_rate)
        indices = np.random.permutation(len(input))

        # Get the Input data pre-processed according with the indexes
        train_input = input[indices[:-num_test]]
        test_input = input[indices[-num_test:]]

        # Get the output data according with the indexes
        train_output = output[indices[:-num_test]]
        test_output = output[indices[-num_test:]]

        return train_input, train_output, test_input, test_output

    def fit_model(
        self, method="Linear Regression", cross_val: int = 5
    ) -> "SupervisedModelOptimization":
        """Model fit of the chosen one by tuning the best possible hyperparameters in a regression problem. If no several parameters introduced, we fit the respective model.

        Args:
            chosen_model (str, optional): Model chosen to perform the data predictions. Defaults to "Linear Regression".
            cross_val (int, optional):  Cross-validation splitting value. Defaults to 5.

        Returns:
            SupervisedModelOptimization: Fitted model.
        """
        if method == "Linear Regression":
            # FIXME: No need for params, remove from here due to the GridSearchCV?
            model_estimator = linear_model.LinearRegression()
            params = None
        elif method == "KNN":
            model_estimator = KNeighborsRegressor()
            params = [
                {
                    "n_neighbors": [1, 2, 3, 4, 5],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                }
            ]
        elif method == "Decision Tree":
            model_estimator = DecisionTreeRegressor()
            # FIXME: Not sure if more criterion should be added
            params = {
                "criterion": ["squared_error", "absolute_error"],
                "max_depth": [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
                "splitter": ["random", "best"],
            }
        elif method == "SVM":
            model_estimator = SVR(kernel="linear")
            # Linear probably the best kernel, since I presume the relations of the data are linear
            # FIXME: Ensure if the other kernels should be here
            # params = [{"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001]}]
            params = [{"C": [1, 10], "gamma": [0.001]}]
        elif method == "Random Forest":
            model_estimator = RandomForestRegressor()
            # FIXME: Not sure if more criterion should be added
            params = [{'n_estimators': [10, 20, 40, 60, 100], 'criterion': ["squared_error", "absolute_error"], 'max_depth':[5,10,20,45,75,100,150]}]
        elif method == "Bagging":
            # Define the base regressor
            base_regressor = DecisionTreeRegressor(random_state=42)

            # Define the Bagging regressor
            model_estimator = BaggingRegressor(base_estimator=base_regressor)
            params = [{'max_samples': [0.2, 0.5, 0.8, 1], 'bootstrap': [True, False]}]

        self.model_estimator = model_estimator

        # If no need for hyper-parameter tuning, we fit the respective model
        if not params:
            self.model = deepcopy(self.model_estimator)
            self.model.fit(self.train_input, self.train_output)
            return self

        # If multiple parameters, we use GridSearchCV for hyper-parameter tuning
        self.model = GridSearchCV(
            estimator=model_estimator,
            param_grid=params,
            scoring="neg_mean_squared_error",
            cv=cross_val,
        )
        self.model.fit(self.train_input, self.train_output)

        # Return own class instance to allow method chaining
        return self

    def predict(self) -> np.ndarray:
        """Method to return the predictions of the fitted model.

        Returns:
            np.ndarray: Model predictions.
        """
        assert self.model, "Train (through the fit function) a certain model"
        self.predictions = self.model.predict(self.test_input)
        return self.predictions

    def score(self) -> float:
        """Method to return the model score.s

        Returns:
            float: Model score.
        """
        return (
            abs(self.model.best_score_)
            if self.model.__class__.__name__ != self.model_estimator.__class__.__name__
            else mean_squared_error(self.test_output, self.predictions)
        )

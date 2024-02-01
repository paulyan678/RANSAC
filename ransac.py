
import numpy as np
from copy import copy

class RANSAC:
    def __init__(self, k, n, model, d, t, supporter_metric, model_error_metric):
        # number of iterations
        self.k = k

        # number of points to randomly sample
        self.n = n

        # model that fits the data
        self.model = model

        # threshold to determine if a point is an supporter
        self.t = t

        # minium number of supporters neede in order for the model to be onsidered
        self.d = d

        # metric to determine if a point is an supporter
        self.supporter_metric = supporter_metric

        # metric to determine how good the model is
        self.model_error_metric = model_error_metric

        self.best_model = None

        self.best_model_error = np.inf

    def fit(self, x_data, y_data):
        iter = 0
        while iter < self.k:
            # randomn select n natural number indices from x_data
            random_indices = np.random.choice(x_data.shape[0], self.n, replace=False)

            x_random_sample = x_data[random_indices]
            y_random_sample = y_data[random_indices]

            model_clone = copy(self.model)

            # fit model to random sample
            model_clone.fit(x_random_sample, y_random_sample)

            # find supporters
            y_random_predicts = model_clone.predict(x_data)
            supporter_errors = self.supporter_metric(y_random_predicts, y_data)
            supporters = np.where(supporter_errors < self.t)[0]

            # if number of supporters is greater than d the we found a good eno8ugh model
            if len(supporters) > self.d:
                # fit model to supporters
                x_supporters = x_data[supporters]
                y_supporters = y_data[supporters]
                model_clone.fit(x_supporters, y_supporters)

                y_supporters_predicts = model_clone.predict(x_supporters)
                model_error = self.model_error_metric(y_supporters_predicts, y_supporters)
                if model_error < self.best_model_error:
                    self.best_model_error = model_error
                    self.best_model = model_clone
            iter += 1
    


     


            



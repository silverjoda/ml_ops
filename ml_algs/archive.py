from sklearn.base import BaseEstimator, ClassifierMixin


class ManualClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, intValue=0, stringParam="defaultValue", otherParam=None):
        self.intValue = intValue
        self.stringParam = stringParam

    def objective(self, X, Y, trial):
        t1 = trial.suggest_float("t1", 0.1, 1)
        t2 = trial.suggest_float("t2", 0.1, 1)
        t3 = trial.suggest_float("t3", 0.1, 1)
        self.params = t1, t2, t3

        batchsize = 100
        indeces_subset = random.sample(range(0, len(X)), batchsize)
        X_subset = X[indeces_subset]
        Y_subset = Y[indeces_subset]

        Y_ = np.array([self._meaning(x) for x in X_subset])

        score = (Y_subset == Y_).sum()

        return score

    def fit(self, X, y=None):
        timeout = 20.0
        print(f"Running optimization for {timeout} seconds...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self.objective(X, y, t), timeout=timeout)

        self.params = [study.best_params['t1'],
                       study.best_params['t2'],
                       study.best_params['t3']]

        return self

    def _meaning(self, x):
        t1, t2, t3 = self.params
        cur_max_val = 0
        cur_min_val = 10
        state = 0
        for i in range(len(x)):
            val = x[i]

            if val > cur_max_val:
                cur_max_val = val
            if val < cur_min_val:
                cur_min_val = val

            if state == 0:
                # In this state the sequence is rising from the start
                if val < cur_max_val - t1:
                    cur_max_val = 0
                    cur_min_val = 10
                    state = 1
            if state == 1:
                # In this state we have already had our failure and pressure is dropping
                if val > cur_min_val + t2:
                    cur_max_val = 0
                    cur_min_val = 10
                    state = 2
            if state == 2:
                # In this state the pressure is rising after the drop and we will just check that it rises above a certain threshold
                if val > cur_min_val + t3:
                    return True
        return False

    def predict(self, X, y=None):
        if len(X.shape) == 1:
            # Filter high frequency noise from sequence
            cumsum_vec = np.cumsum(np.insert(X, 0, 0))
            seq_filt = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

            return self._meaning(seq_filt)
        else:
            cumsum_vec = np.cumsum(X, axis=1)
            seq_filt = (cumsum_vec[:, window_width:] - cumsum_vec[:, :-window_width]) / window_width
            res = [self._meaning(x) for x in seq_filt]

            return res

    def score(self, X, y=None):
        return (sum(self.predict(X) == y))


clf = ManualClassifier()
clf.fit(X_train, y_train)

y_hat = clf.predict(X_train)
print("F1 score on trn", sklearn.metrics.f1_score(y_train, y_hat))

y_hat_tst = clf.predict(X_test)
print("F1 score on tst", sklearn.metrics.f1_score(y_test, y_hat_tst))

display_cm = sklearn.metrics.ConfusionMatrixDisplay.from_estimator(
    clf,
    X_test,
    y_test,
    display_labels=["normal", "failed"],
    cmap=plt.cm.Blues,
)

_ = display_cm.ax_.set_title("2-class Confusion matrix")
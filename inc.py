import numpy as np
import pandas as pd
import pygad
import math
import glob
import pickle
from itertools import product
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean


n_jobs = 15


def calc_lags(data, cols):
    to_concat = [data]

    for i in range(1, 8):
        lagged = data.loc[:, cols].shift(periods=i)
        lagged.columns = [f"{col}_L{i}" for col in lagged.columns]

        to_concat.append(lagged)

    return pd.concat(to_concat, axis=1)


def calc_wma(data, cols, period):
    wma = (
        data.loc[:, cols]
        .rolling(period)
        .apply(
            lambda x: np.sum(np.arange(1, period + 1) * x)
            / np.sum(np.arange(1, period + 1))
        )
    )
    wma = wma.shift()
    wma.columns = [f"{col}_WMA30" for col in wma.columns]

    return pd.concat((data, wma), axis=1)


def genetic_filter(X, y):
    def f(i, j):
        mi = mutual_info_regression(i.reshape(-1, 1), j)[0]
        corr, _ = pearsonr(i, j)

        return mi + abs(corr)

    f_values = {}
    ncols = X.shape[1]

    results = Parallel(n_jobs=n_jobs)(
        delayed(f)(X[:, i], X[:, j])
        for i in range(ncols - 1)
        for j in range(i + 1, ncols)
    )

    index = 0
    for i in range(ncols - 1):
        for j in range(i + 1, ncols):
            f_values[(i, j)] = results[index]
            f_values[(j, i)] = results[index]
            index += 1

    results = Parallel(n_jobs=n_jobs)(delayed(f)(X[:, i], y) for i in range(ncols))

    index = 0
    for i in range(ncols):
        f_values[("target", i)] = results[index]
        index += 1

    def fitness_func(solution, solution_idx):
        idx_selected = np.nonzero(solution)[0]

        f_features_target = 0
        for idx in idx_selected:
            m = f_values[("target", idx)]
            f_features_target += m

        f_features = 0
        count = 0
        for i in range(len(idx_selected) - 1):
            for j in range(i + 1, len(idx_selected)):
                count += 1
                m = f_values[(idx_selected[i], idx_selected[j])]
                f_features += m

        return f_features_target - f_features

    ga = pygad.GA(
        num_parents_mating=4,
        keep_parents=3,
        sol_per_pop=100,
        num_generations=1000,
        num_genes=ncols,
        crossover_type="two_points",
        mutation_type="random",
        mutation_probability=0.001,
        parent_selection_type="rws",
        gene_space=(0, 1),
        fitness_func=fitness_func,
        parallel_processing=["thread", n_jobs],
    )

    ga.run()

    return np.nonzero(ga.best_solution()[0])[0], ga


class GeneticWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model, validation_size=0.2):
        self.model = clone(model)
        self.val_size = validation_size

    def fit(self, X, y):
        separator = math.floor(len(y) * (1 - self.val_size))
        X_train = X[:separator, :]
        y_train = y[:separator]
        X_test = X[separator:, :]
        y_test = y[separator:]

        def fitness_func(solution, solution_idx):
            idx_selected = np.nonzero(solution)[0]

            model = clone(self.model)

            accs = []
            for _ in range(5):
                model.fit(X_train[:, idx_selected], y_train)
                y_pred = model.predict(X_test[:, idx_selected])
                accs.append(accuracy_score(y_test, y_pred))

            return np.mean(accs)

        self.ga = pygad.GA(
            num_parents_mating=4,
            keep_parents=3,
            sol_per_pop=100,
            num_generations=30,
            num_genes=X_train.shape[1],
            crossover_type="two_points",
            mutation_type="random",
            mutation_probability=0.001,
            parent_selection_type="rws",
            gene_space=(0, 1),
            fitness_func=fitness_func,
            random_seed=15,
            parallel_processing=["thread", n_jobs],
            stop_criteria=["saturate_15"],
        )

        self.ga.run()
        self.selected = np.nonzero(self.ga.best_solution()[0])[0]

        return self

    def transform(self, X):
        if not self.selected:
            raise Exception("You must run the fit method first")

        return X[:, self.selected].copy()

    def get_ga(self):
        return self.ga

    def get_selected(self):
        return self.selected


class Ensemble1(BaseEstimator, ClassifierMixin):
    def __init__(self, mlp_hidden_layers_sizes=5, epochs=500):
        self.mlp_hidden_layers_sizes = mlp_hidden_layers_sizes
        self.epochs = epochs

        self.regressor = MLPRegressor(
            hidden_layer_sizes=mlp_hidden_layers_sizes,
            solver="adam",
            max_iter=epochs,
            activation="tanh",
            learning_rate="adaptive",
        )

        self.tree_classifier = DecisionTreeClassifier()

        self.epochs = epochs

    def fit(self, X, y):
        self.regressor.fit(X, y)
        y_reg_pred = self.regressor.predict(X)
        self.tree_classifier.fit(y_reg_pred.reshape(-1, 1), y)

        return self

    def predict(self, X):
        y_pred = self.regressor.predict(X)
        return self.tree_classifier.predict(y_pred.reshape(-1, 1))


class Ensemble2(BaseEstimator, ClassifierMixin):
    def __init__(self, mlp_hidden_layers_sizes=5, epochs=500, n_clusters=3):
        self.mlp_hidden_layers_sizes = mlp_hidden_layers_sizes
        self.epochs = epochs
        self.n_clusters = n_clusters

        self.clusterizer = KMeans(n_clusters=n_clusters)

        self.regressor = MLPRegressor(
            hidden_layer_sizes=mlp_hidden_layers_sizes,
            solver="adam",
            max_iter=epochs,
            activation="tanh",
            learning_rate="adaptive",
        )

        self.tree_classifier = DecisionTreeClassifier()

        self.epochs = epochs

    def fit(self, X, y):
        self.clusterizer.fit(X)
        _X = np.append(X, self.clusterizer.labels_.reshape(-1, 1), axis=1)
        self.regressor.fit(_X, y)
        y_reg_pred = self.regressor.predict(_X)
        self.tree_classifier.fit(y_reg_pred.reshape(-1, 1), y)

        return self

    def predict(self, X):
        labels = self.clusterizer.predict(X)
        _X = np.append(X, labels.reshape(-1, 1), axis=1)
        y_pred = self.regressor.predict(_X)
        return self.tree_classifier.predict(y_pred.reshape(-1, 1))


def train_test_split(X, y, test_size=0.2):
    nrows = X.shape[0]
    sep = round(nrows * (1 - test_size))

    return X[:sep, :], X[sep:, :], y[:sep], y[sep:]


def make_predictions(model, X_train, y_train, X_test):
    import warnings

    warnings.simplefilter("ignore")

    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_tests(model, X_train, y_train, X_test, y_test, test_runs=1):
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(make_predictions)(model, X_train, y_train, X_test)
        for i in range(test_runs)
    )

    def metrics(y_true, y_pred):
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, y_pred)

    aucs = []
    accs = []
    for preds in predictions:
        m = metrics(y_test, preds)
        aucs.append(m[0])
        accs.append(m[1])

    aucs_mu = round(np.mean(aucs), 2)
    aucs_std = round(np.std(aucs), 2)
    accs_mu = round(np.mean(accs) * 100, 2)
    accs_std = round(np.std(accs) * 100, 2)

    model_name = (
        list(dict(model.named_steps).values())[-1]
        if isinstance(model, Pipeline)
        else type(model).__name__
    )

    return [model_name, aucs_mu, aucs_std, accs_mu, accs_std]


def find_best_params(base_model, X_train, y_train, X_test, y_test, param_grid, runs=5):
    def eval(model, X_train, y_train, X_test, y_test):
        m = clone(model)
        m.fit(X_train, y_train)

        return accuracy_score(y_test, m.predict(X_test))

    combinations = product(*list(param_grid.values()))
    best_params = None
    best_score = None

    for combination in combinations:
        params = dict(zip(list(param_grid.keys()), combination))
        accs = Parallel(n_jobs=min(runs, n_jobs))(
            delayed(eval)(
                clone(base_model).set_params(**params), X_train, y_train, X_test, y_test
            )
            for _ in range(runs)
        )

        acc_mu = np.mean(accs)

        if best_params is None or acc_mu > best_score:
            best_params = params
            best_score = acc_mu

    return best_params, best_score


def dump_results(results, filename):
    with open(f"experiments/{filename}", "wb") as f:
        pickle.dump(results, f)


def load_results():
    results = []
    files = glob.glob("experiments/**.bin")

    for file in files:
        with open(file, "rb") as f:
            results.extend(pickle.load(f))

    return results


def CCSA(D, pop_size, fitness, max_iter, max_saturation=0):
    lu = (-1 * np.ones(D), 1 * np.ones(D))
    X = lu[0] + np.random.rand(pop_size, D) * (
        np.tile(lu[1] - lu[0], pop_size).reshape(pop_size, D)
    )
    m = X.copy()
    f = np.array([fitness(X[i, :]) for i in range(pop_size)])
    f_best = f.copy()
    m_g_f_best_index = np.argmin(f_best)
    m_g_f_best = f_best[m_g_f_best_index]
    m_g_best_pos = m[m_g_f_best_index, :]
    f_best_history = [m_g_f_best]

    for t in range(1, max_iter):
        fl = 2.02 - t * ((1.08) / max_iter)

        for i in range(pop_size):

            # Generating neighborhood
            out = []
            alpha = 0.02
            for k in range(pop_size):
                if k != i:
                    W = (alpha + (f[i] - f_best[k])) / np.sum(np.abs(f[i] - f_best[k]))
                    out.append(euclidean(X[i, :], m[k, :]) * W)

            out = np.array(out)
            mu = np.mean(out)
            neigh = np.argwhere(out < mu).reshape(1, -1)[0]
            non_neigh = np.argwhere(out > mu).reshape(1, -1)[0]

            Local = 0
            if len(neigh) != 0:
                Local = neigh[np.random.randint(len(neigh))]

            Global = 0
            if len(non_neigh) != 0:
                Global = np.argmin(f_best[non_neigh])

            if f_best[Local] < f_best[Global]:
                # NLS Strategy
                X[i, :] = X[i, :] + fl * np.random.rand(1, D) * (m[Local, :] - X[i, :])
            else:
                # NGS Strategy
                X[i, :] = np.random.rand(1, D) * fl * (m[Global, :] - X[i, :])

            X[i, :] = (
                ((X[i, :] >= lu[0]) & (X[i, :] <= lu[1])) * X[i, :]
                + (X[i, :] < lu[0])
                * (lu[0] + 0.25 * (lu[1] - lu[0]) * np.random.rand(D))
                + (X[i, :] > lu[1])
                * (lu[1] - 0.25 * (lu[1] - lu[0]) * np.random.rand(D))
            )

            f[i] = fitness(X[i, :])

            if f_best[i] < f[i]:
                nstep = np.fix(np.random.rand(1) * D).astype(int)[0]
                r0 = len([k for k in range(pop_size) if k != i])
                X_r = X[np.random.randint(r0), :]
                Nj = np.random.randint(50)

                for j in range(Nj):
                    wasfl = 2.02 - j * (1.08 / Nj)
                    k = np.random.permutation(len(X[i, :]))[:nstep]
                    Tmp = m[i, :]
                    Tmp[k] = m_g_best_pos[k] + np.random.rand(len(k)) * wasfl * (
                        X_r[k] - X[i, k]
                    )
                    Tmp = (
                        ((Tmp >= lu[0]) & (Tmp <= lu[1])) * Tmp
                        + (Tmp < lu[0])
                        * (lu[0] + 0.25 * (lu[1] - lu[0]) * np.random.rand(len(Tmp)))
                        + (Tmp > lu[1])
                        * (lu[1] - 0.25 * (lu[1] - lu[0]) * np.random.rand(len(Tmp)))
                    )

                    f_Tmp = fitness(Tmp)

                    if f_Tmp < f[i]:
                        f[i] = f_Tmp
                        X[i, :] = Tmp

                    if f_Tmp < f_best[i]:
                        f_best[i] = f_Tmp
                        m[i, :] = Tmp

            if f[i] < f_best[i]:
                m[i, :] = X[i, :]
                f_best[i] = f[i]

        m_g_f_best_index = np.argmin(f_best)
        m_g_f_best = f_best[m_g_f_best_index]
        m_g_best_pos = m[m_g_f_best_index, :]
        f_best_history.append(m_g_f_best)

        if (max_saturation > 0) and (
            len(np.unique(np.array(f_best_history[-1 * max_saturation :]))) == 1
        ):
            break

    return m_g_best_pos, f_best_history

import numpy as np
import pandas as pd

EUCLIDIAN = 0
MANHATTAN = 1
COSINE = 2

class KNeighborsClassifier:
    def __init__(self, n_neighbors, metric):
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.feature_names_in_ = None
        self.x = None
        self.y = None

    @property
    def classes_(self):
        if type(self.y) == np.ndarray:
            classes = list(set(self.y))
            return np.array(classes)
        else:
            raise ValueError("Must call fit first to load data")
        
    @property
    def n_samples_fit_(self):
        if type(self.x) == np.ndarray:
            return self.x.shape[0]
        else:
            raise ValueError("Must call fit first to load data")
        

    def fit(self, x, y):
        self.x = x
        self.y = y

        if type(self.x) == pd.DataFrame:
            self.feature_names_in_ = x.columns
            self.x = np.array(self.x)
        
        if type(self.y) == pd.Series:
            self.y = np.array(self.y)
        
        return self

    # Old version of function, use vectorized version
    # def kneighbors_old(self, X, return_distances=True):
    #     n_neighbors = self.n_neighbors

    #     distances = []

    #     for i in range(len(self.x)):
    #         sample = self.x[i]
    #         distances.append((i, self.get_distance(sample, X)))

    #     together = sorted(distances, key=lambda x: x[1])
        
    #     indices = [i[0] for i in together[:n_neighbors]]
    #     if return_distances:
    #         dist = [i[1] for i in together[:n_neighbors]]
    #         return (indices, dist)
    #     else:
    #         return indices
        
    # Vectorized version
    def kneighbors(self, X, return_distances=True):
        n_neighbors = self.n_neighbors

        if self.metric == EUCLIDIAN:
            distances = np.linalg.norm(self.x - X, axis=1)
        elif self.metric == MANHATTAN:
            distances = np.sum(np.abs(self.x - X), axis=1)
        elif self.metric == COSINE:
            dots = np.dot(self.x, X)
            x_norms = np.linalg.norm(self.x, axis=1)
            sample_norm = np.linalg.norm(X)
            distances = 1 - (dots / (x_norms*sample_norm))
        else:
            raise ValueError("Metric must be 0, 1, or 2") 
        
        zipped_distances = zip( list(range(self.x.shape[0])), distances)
        together = sorted(zipped_distances, key=lambda x: x[1])[:n_neighbors]

        indices = np.array([i[0] for i in together])
        if return_distances:
            dist = [i[1] for i in together]
            return (indices, dist)
        else:
            return indices
        

    def predict(self, samples_in):
        labels =  self.y
        output_labels = []

        for sample in samples_in:
            nearest_indices, dist = self.kneighbors(sample, True)
            
            # Weighs distance by repeating indices 0-k times
            # The smaller the distance, the more it's repeated
            max_dist = np.max(dist)
            repeat_counts = np.int64( (max_dist - dist) / max_dist * self.n_neighbors)

            # Credit: ChatGPT vectorizing this
            expanded = np.repeat(nearest_indices, repeat_counts)
            nearest_indices = np.concatenate([nearest_indices, expanded])

            closest_labels = labels[nearest_indices]#np.array([labels[i] for i in nearest_indices]).astype('int64')

            # Credit geeksforgeeks.org
            y = np.bincount(closest_labels) 
            index = np.argmax(y)
            output_labels.append(index)

        return np.array(output_labels )

    def score(self, X, y):
        # Gets % accuracy
        predictions = self.predict(X)
        labels = y
        return np.sum(predictions == labels)/X.shape[0]
        


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # NOTE: If adding features, make sure to add feature column in the dataframe (in this examples)
    X, y = make_blobs(n_samples=200, centers=3, n_features=3, random_state=1)

    # Split testing data and example data
    X_test, y_test = X[150:], y[150:]
    X, y = X[:150], y[:150]


    knn = KNeighborsClassifier(n_neighbors=7, metric=MANHATTAN)

    # Turning blobs into DataFrame for testing
    X_as_df = pd.DataFrame({"x":X[:,0], "y":X[:,1], "z":X[:,2]})
    y_as_series = pd.DataFrame({"y":y})["y"]

    knn.fit(X_as_df, y_as_series)
    print(f"Classes: {knn.classes_}")
    print(f"Feature Names: {list(knn.feature_names_in_)}")

    y_pred = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    print(f"Score: {score}")

    print("\nUsing 3D data but only plotting X and Y!")

    plt.scatter(x=X[:,0], y=X[:,1], c=y, marker='o')
    plt.scatter(x=X_test[:,0], y=X_test[:,1], c=y_pred, marker='+')
    plt.show()
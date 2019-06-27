__author__ = 'mikhail91'

import numpy
from sklearn.cluster import DBSCAN

class Clusterer(object):

    def __init__(self, cluster=DBSCAN(eps=0.06)):

        self.cluster = cluster

    def get_polar(self, x, y):
        """
        Calculate hits phi coordinates in polar system.

        Parameters
        ----------
        x : array-like
            X-coordinates of hits.
        y : array-like
            Y-coordinates of hits.

        Returns
        -------
        phi : array-like
            Phi coordinates of hits.
        """

        x = numpy.array(x)
        y = numpy.array(y)

        phi = numpy.arctan(y / x) * (x != 0) + numpy.pi * (x < 0) + 0.5 * numpy.pi * (x==0) * (y>0) + 1.5 * numpy.pi * (x==0) * (y<0)
        r = numpy.sqrt(x**2 + y**2)

        return r, phi


    def splitter(self, labels, X):
        """
        Separate two close tracks.

        Parameters
        ----------
        labels : array-like
            Recognized hit labels.
        X : ndarray-like
            Hit features.

        Returns
        ------
        labels : array-like
            New recognized hit labels.
        """

        x, y, layer = X[:, 2], X[:, 3], X[:, 0]
        r, phi = self.get_polar(x, y)

        ind = numpy.arange(len(X))
        unique_labels = numpy.unique(labels[labels != -1])
        if len(unique_labels) == 0:
            return labels
        track_id = unique_labels[-1] + 1
        #print unique_labels

        for lab in unique_labels:

            track_ind = ind[labels == lab]

            track_layer = layer[track_ind]
            track_phi = phi[track_ind]

            track1 = []
            track2 = []

            for l in numpy.unique(track_layer):

                ind_layer = track_ind[track_layer == l]
                phi_layer = track_phi[track_layer == l]

                hit_loc_ind = numpy.argsort(phi_layer)

                if len(hit_loc_ind) == 0:
                    continue

                track1.append(ind_layer[hit_loc_ind[0]])

                if len(ind_layer) > 1:

                    track2.append(ind_layer[hit_loc_ind[-1]])

            if len(track2)>=2:
                labels[track2] = track_id
                track_id += 1

        return labels

    def fit(self, X, y):
        pass

    def predict_single_event(self, X):

        x, y = X[:, 2], X[:, 3]
        r, phi = self.get_polar(x, y)

        self.cluster.fit(phi.reshape(-1, 1))

        labels = self.cluster.labels_

        labels = self.splitter(labels, X)

        return labels

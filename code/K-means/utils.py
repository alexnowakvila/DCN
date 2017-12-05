import numpy as np
import os
from sklearn.cluster import KMeans
import pdb


def Lloyds(input, n_clusters=8):
  kmeans = KMeans(n_clusters=n_clusters)
  nb_pbs, nb_samples, d = input.shape
  Costs = []
  for i in range(nb_pbs):
    inp = input[i]
    labels = kmeans.fit_predict(inp)
    cost = 0
    for cl in range(n_clusters):
      ind = np.where(labels==cl)[0]
      if ind.shape[0] > 0:
        x = inp[ind]
        mean = x.mean(axis=0)
        cost += np.mean(np.sum((x - mean)**2, axis=1), axis=0)*ind.shape[0]
      # cost += np.var(inp[ind], axis=0)*ind.shape[0]
    Costs.append(cost)
  Cost = sum(Costs)/len(Costs)
  return Cost

def cost(input, e, n_clusters=8):
  kmeans = KMeans(n_clusters=n_clusters)
  nb_pbs, nb_samples, d = input.shape
  Costs = []
  for i in range(nb_pbs):
    inp = input[i]
    labels = e[i]
    cost = 0
    for cl in range(n_clusters):
      ind = np.where(labels==cl)[0]
      if ind.shape[0] > 0:
        x = inp[ind]
        mean = x.mean(axis=0)
        cost += np.mean(np.sum((x - mean)**2, axis=1), axis=0)*ind.shape[0]
        if np.isnan(cost):
          pdb.set_trace()
      # cost += np.var(inp[ind], axis=0)*ind.shape[0]
    Costs.append(cost)
  Cost = sum(Costs)/len(Costs)
  return Cost



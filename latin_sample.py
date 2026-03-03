import scipy as sp

sampler = sp.stats.qmc.LatinHypercube(4)
sample = sampler.random(n=1)
scaled_sample = sp.stats.qmc.scale(sample, [50, 0, 1, 2], [300, 1, 2, 5])
print(scaled_sample)

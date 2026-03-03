import scipy as sp

sampler = sp.stats.qmc.LatinHypercube(4, strength=2)
sample = sampler.random(n=9)
scaled_sample = sp.stats.qmc.scale(sample, [50, 0, 1, 2], [300, 1, 2, 5])
print(scaled_sample)


# R code

library(MASS)
counts <- read.csv("counts.csv", header = TRUE, stringsAsFactors = FALSE)

Ntotal = length(counts[, 1])
Nfit = round(0.5 * Ntotal)
print(c(Ntotal, Nfit))

print(c(mean(counts[, 2]), sd(counts[, 2])))

pars = fitdistr(counts[, 2], "negative binomial")$estimate
print(pars)

# calc by sampling

# their code uses rate=1... why?
sample = sort(rgamma(Ntotal, shape=pars[1], rate=pars[1]/pars[2]), decreasing=T) #/pars[2])

print(c(mean(sample), sd(sample)))

conc = sum(sample[1:Nfit]) / sum(sample)

print(conc)

nsample = sample / sum(sample)

gini = 1 / Ntotal * (2 * sum(cumsum(nsample)) - Ntotal - 1)
print(gini)

# calc by integration

p = seq(0.025, 0.975, by=0.05)
Fp = numeric(length(p))
for (i in 1:length(p)) {
  Fp[i] = qgamma(seq(0.0001, 1.0 - p[i], by=0.0001), shape=pars[1], rate=1) * 0.0001 / pars[1]
}
gini = 2 * sum(Fp) * 0.05 - 1
print(gini)
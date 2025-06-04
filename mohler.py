# Converted R code from Mohler (2019)


import numpy as np
import numpy.typing as npt
import scipy.stats


def simple_nbinom_fit(sample: npt.NDArray) -> tuple[float, float]:
    mu = np.mean(sample)
    sigma_sqr = np.var(sample)

    n = mu**2 / (sigma_sqr - mu)
    p = mu / sigma_sqr
    return n, p


def test_simple_nbinom_fit() -> None:
    # Generate sample data
    n = 2
    p = 0.9
    sample = scipy.stats.nbinom.rvs(n=n, p=p, size=1000)

    n_est, p_est = simple_nbinom_fit(sample)
    print(n_est / n - 1, p_est - p)


# --- Placeholder Inputs ---
# Replace with your actual data
# counts: vector of counts in each cell (e.g., observed frequencies)
# Ntotal: total number of cells (length of the counts vector)
# Nflag: number of cells used as the top p fraction (e.g., top 10%)

# Example placeholder values (replace with your actual data)
counts = np.random.randint(0, 100, size=1000)  # Example: 1000 cells with counts
Ntotal = len(counts)
Nflag = int(Ntotal * 0.1)  # Example: Top 10%

print("Using example data:")
print(f"  Ntotal: {Ntotal}")
print(f"  Nflag: {Nflag}")
print(f"  First 10 counts: {counts[:10]}")
print("-" * 20)


# --- Fig. 7 R code for Poisson-Gamma estimator via simulation ---
# Inputs:
# counts: vector of counts in each cell
# Ntotal: total number of cells
# Nflag: number of cells used as the top p fraction


def simulation() -> None:
    print("--- Simulation Method ---")

    # library(MASS) -> Handled by importing scipy.stats

    # pars=fitdistr(counts, "negative.binomial")$estimate #estimate parameters mu, k
    # In scipy, nbinom.fit estimates n (size/k) and p (prob).
    # The R code uses pars[1] as the shape for the Gamma distribution.
    # In the Poisson-Gamma mixture model where Poisson rate ~ Gamma(shape=k, rate=lambda),
    # the resulting variable is Negative Binomial(size=k, mu=k/lambda).
    # If rate=1 in the Gamma, then mu=k. fitdistr for negbin estimates mu and size (k).
    # The R code using shape=pars[1], rate=1 suggests pars[1] is the estimated size (k).
    # scipy.stats.nbinom.fit returns (n, p), where n is the size (k).
    try:
        # Ensure counts are integers as expected for Negative Binomial fit
        counts_int = counts.astype(int)
        # nbinom.fit returns (n, p), where n is the 'size' parameter (R's k)
        # nbinom.fit is not implemented - see https://stackoverflow.com/questions/23816522/fitting-negative-binomial-in-python
        estimated_size, p_est = simple_nbinom_fit(counts_int)  # scipy.stats.nbinom.fit(counts_int)
        print(f"Estimated Negative Binomial size (k): {estimated_size:.4f}")

        # simulated_gam=rgamma(Ntotal, shape=pars[1], rate=1)
        # scipy.stats.gamma uses shape (a) and scale (1/rate)
        simulated_gam = scipy.stats.gamma.rvs(a=estimated_size, scale=1, size=Ntotal)

        # sorted_gam=sort(simulated_gam, decreasing=T)
        sorted_gam = np.sort(simulated_gam)[::-1]  # Sort descending

        # concentration=sum(sorted_gam[1:Nflag])/sum(sorted_gam)
        # Python slicing is 0-based and exclusive of the end index.
        # R's 1:Nflag corresponds to indices 0 to Nflag-1 in Python, which is [:Nflag]
        concentration_sim = np.sum(sorted_gam[:Nflag]) / np.sum(sorted_gam)
        print(f"Concentration (simulated): {concentration_sim:.4f}")

        # normalized_gam=sorted_gam/sum(sorted_gam)
        normalized_gam = sorted_gam / np.sum(sorted_gam)

        # gini=(1/Ntotal)*(2*sum(cumsum(normalized_gam))-Ntotal-1)
        gini_sim = (1.0 / Ntotal) * (2 * np.sum(np.cumsum(normalized_gam)) - Ntotal - 1)
        print(f"Gini coefficient (simulated): {gini_sim:.4f}")

    except Exception as e:
        print(f"Error during simulation: {e}")
        print("Ensure 'counts' data is suitable for Negative Binomial fitting.")
        raise

    print("-" * 20)


# --- Fig. 8 R code for Poisson-Gamma estimator via numerical integration ---
# Inputs:
# counts: vector of counts in each cell
# Ntotal: total number of cells
# Nflag: number of cells used as the top p fraction


def integration() -> None:
    print("--- Numerical Integration Method ---")

    # library(MASS) -> Handled by importing scipy.stats

    # pars=fitdistr(counts, "negative.binomial")$estimate #estimate parameters mu, k
    # Same fitting as in the simulation method
    try:
        counts_int = counts.astype(int)
        # nbinom.fit returns (n, p), where n is the 'size' parameter (R's k)
        estimated_size, p_est = simple_nbinom_fit(counts_int)  # scipy.stats.nbinom.fit(counts_int)
        print(f"Estimated Negative Binomial size (k): {estimated_size:.4f}")

        # concentration=1-sum(qgamma(seq(.0001,1-Nflag/Ntotal,by=.0001), shape=pars[1], rate=1))*.0001/pars[1]
        # R's seq(start, end, by=step) includes the endpoint if exactly reachable.
        # np.arange(start, end, step) excludes the endpoint. Add a small epsilon or half step to ensure inclusion.
        step_size = 0.0001
        upper_limit_q = 1.0 - Nflag / Ntotal
        q_probs = np.arange(step_size, upper_limit_q + step_size / 2.0, step_size)  # Use half step trick

        # qgamma(..., shape=pars[1], rate=1) -> scipy.stats.gamma.ppf(..., a=estimated_size, scale=1)
        q_values = scipy.stats.gamma.ppf(q_probs, a=estimated_size, scale=1)

        # Numerical integration approximation: sum(values) * step_size
        integral_approx_conc = np.sum(q_values) * step_size

        # The R formula is 1 - (Integral_0^(1-p) Q(q) dq) / mean(X)
        # where p = Nflag/Ntotal, Q(q) is the quantile function, mean(X) = shape/rate = estimated_size/1
        concentration_num = 1.0 - integral_approx_conc / estimated_size
        print(f"Concentration (numerical): {concentration_num:.4f}")

        # p=seq(.025,.975,by=.05)
        gini_p_seq = np.arange(0.025, 0.975 + 0.05 / 2.0, 0.05)  # Use half step trick

        # Fp=numeric(length(p))
        Fp = []  # Use a list to store results

        # for(i in 1:length(p)){Fp[i]=qgamma(seq(.0001,1-p[i],by=.0001), shape=pars[1], rate=1))*.0001/pars[1]}
        # Loop through the gini_p_seq (p values)
        step_size_inner = 0.0001
        for p_val in gini_p_seq:
            # seq(.0001, 1-p[i], by=.0001)
            inner_upper_limit_q = 1.0 - p_val
            inner_q_probs = np.arange(
                step_size_inner,
                inner_upper_limit_q + step_size_inner / 2.0,
                step_size_inner,
            )

            # qgamma(...)
            inner_q_values = scipy.stats.gamma.ppf(inner_q_probs, a=estimated_size, scale=1)

            # Numerical integration approx
            inner_integral_approx = np.sum(inner_q_values) * step_size_inner

            # /pars[1] -> /estimated_size
            # This inner calculation is L(1 - p_val) using numerical integration
            L_1_minus_p_val = inner_integral_approx / estimated_size

            Fp.append(L_1_minus_p_val)

        Fp = np.array(Fp)  # Convert list to numpy array

        # gini=2*sum(Fp)*.05-1
        # sum(Fp)*.05 is numerical integration of L(1-p) dp over the range of p
        # from 0.025 to 0.975.
        gini_num = 2 * np.sum(Fp) * 0.05 - 1
        print(f"Gini coefficient (numerical): {gini_num:.4f}")

    except Exception as e:
        print(f"Error during numerical integration: {e}")
        print("Ensure 'counts' data is suitable for Negative Binomial fitting.")
        raise

    print("-" * 20)


"""
Explanation of the Conversion:

1.  **Library:** R's `library(MASS)` providing `fitdistr` is replaced by importing specific functions from `scipy.stats`
 (for distribution fitting and functions) and `numpy` (for array operations, sorting, summing, cumulative summing, and
 sequences).

2.  **Input Variables:** The R code assumes `counts`, `Ntotal`, and `Nflag` are defined. Placeholders are added in
Python.

3.  **`fitdistr`:** R's `fitdistr(counts, \"negative.binomial\")` is replaced by `scipy.stats.nbinom.fit(counts)`.
This function estimates the parameters of the negative binomial distribution. `scipy.stats.nbinom` uses `n` (size/k)
and `p` (probability). `fitdistr` usually estimates `mu` (mean) and `size` (k). Based on the Gamma parameters used
later (`shape=pars[1], rate=1`), `pars[1]` is interpreted as the estimated `size` (k). `scipy.stats.nbinom.fit`
returns `(n, p)`, so we take the first element `[0]` which is the estimated `n` (size).

4.  **`rgamma`:** R's random number generator `rgamma(n, shape, rate)` is replaced by
`scipy.stats.gamma.rvs(a=shape, scale=1/rate, size=n)`. The R code uses `rate=1`, so Python uses `scale=1`. The shape
 parameter `pars[1]` becomes `a=estimated_size`.

5.  **`sort(..., decreasing=T)`:** R's descending sort is replaced by `numpy.sort(...)[::-1]`.

6.  **`sum(...)`:** R's `sum` is replaced by `numpy.sum(...)`.

7.  **Indexing and Slicing:** R's 1-based inclusive indexing (`1:Nflag`) is replaced by Python's 0-based half-open
slicing (`[:Nflag]`).

8.  **`cumsum(...)`:** R's `cumsum` is replaced by `numpy.cumsum(...)`.

9.  **`seq(start, end, by=step)`:** R's sequence generation is replaced by `numpy.arange(start, end, step)`. Note that
`arange` *excludes* the end point, while `seq` includes it if reachable. A common Python pattern to match `seq` is
`np.arange(start, end + step_size/2.0, step_size)`. This is used for both sequences in the numerical integration part.

10. **`qgamma(...)`:** R's quantile function (inverse CDF) `qgamma(p, shape, rate)` is replaced by
`scipy.stats.gamma.ppf(p, a=shape, scale=1/rate)`. Again, `rate=1` in R means `scale=1` in Python.

11. **`numeric(length)`:** R's pre-allocation is replaced by creating an empty list and appending results in the loop,
then converting to a NumPy array if needed (though summing a list works too). A list comprehension could also be used
for a more concise loop.

12. **Loops:** The R `for (i in 1:length(p))` loop is translated to a standard Python `for p_val in p_seq:`
loop iterating directly over the sequence elements.\n\nThis conversion aims to be a direct translation of the
operations performed in the original R code snippets.

"""

if __name__ == "__main__":
    # test_simple_fit()
    simulation()
    integration()

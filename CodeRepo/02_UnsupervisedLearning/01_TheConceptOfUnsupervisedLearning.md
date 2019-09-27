# Unsupervised Learning

## Overview

- Clustering (k-means)
- Mixture of Gaussians
- Jensen’s Inequality
- EM (Expectation-Maximization)

## Clustering

### Application

- Biology: clustering of genes
- Market sales
- News, etc.

### K-Means Clustering Algorithm

Input: ${x^{(1)}, x^{(2)}, \cdots, x^{(m)}}$.

#### Algorithm:

1. Initialization (cluster centroids): $\mu_1, \mu_2, \cdots, \mu_k \in \mathbb{R}^n$.
2. Repeat until convergence,
   1. Assign observations to closest cluster center: $c_m^{(i)} \gets arg \underset{j}{min} || x^{i} - \mu_j||_2$,
   2. Shift cluster centroid: $\mu_j \gets \frac{\Sigma_{i=1}^m 1\{c^{(i)} = j\}x^{(i)}}{\Sigma_{i=1}^m 1\{c^{(i)} = j\}}$.

#### Convergence Analysis

Define **distortion function**: $J(c,\mu) = \Sigma_{i=1}^{m} || x^{(i)} - \mu_{c^{(i)}} ||^2_2$, we can prove that the K-Means is a coordinate descent on $J$. Specifically,

##### K-Means as a coordinate descent algorithm

Let’s look at the repetitive update step:

1. Assign observations to closet cluster center: $c_m^{(i)} \gets arg \underset{j}{min} {|| x^{i} - \mu_j||}_2$

2. Revise cluster centers as mean of assigned observations: 

   $$\mu_j \gets \frac{\Sigma_{i=1}^m 1\{c^{(i)} = j\}x^{(i)}}{\Sigma_{i=1}^m 1\{c^{(i)} = j\}} \iff \mu_j \gets arg \underset{\mu}{min} \Sigma_{i:c^{(i)}_m = j} {||\mu - x^{(i)}||}_2^2$$

We can tell that here we are using an alternating minimization scheme:

1. c given $\mu$
2. $\mu$ given c

$\implies$ **coordinate descent**

Note that the “true” right answer for the number of cluster is often ambiguous. And K-Means is susceptible to local minima, since the distortion function is not a convex function. K-Means usually takes $2-norm$s.

#### Density Estimation

Estimate density to detect outliers:

$$\{x^{(1)}, x^{(2)}, \cdots, x^{(m)}\} \sim P(x)$$

Anomaly detection (aircraft manufacture, credit card unusual transactions, etc.)

There’s a latent (hidden/unobserved) random variable $z$, and $x^{(i)}, z^{(i)}$ have a joint distribution:

$$P(x^{(i)}, z^{(i)}) = P(x^{(i)}|z^{(i)}) P(z^{(i)})$$,

where $z^{(i)} \sim Multinomial(\phi)$ ($Bernoulli$ for 2 cases), $\phi_j \geq 0, \Sigma_j \phi_j = 1$, $x^{(i)} | z^{(i)} = j \sim \mathcal{N}(\mu_j, \Sigma_j)$.

If we knew $z^{(i)}$’s, we have, from Maximum Likelihood Estimation, we have the log-likelihood,

$$\begin{aligned}l(\phi, \mu, \Sigma) &= \Sigma_{i=1}^m \log p(x^{(i)}, z^{(i)}; \phi, \mu, \Sigma) \\ &= \Sigma_{i=1}^m \log \Sigma_{z^{(i)} = 1}^k p(x^{(i)}|z^{(i)};\mu, \Sigma)p(z^{(i)};\phi)\\ &= \Sigma_{i=1}^m \log p(x^{(i)}|z^{(i)};\mu,\Sigma) + \log p(z^{(i)};\phi).\end{aligned}$$

Maximizing this w.r.t. $\phi, \mu, \Sigma$, we have:

$$\begin{aligned}\phi_j &= \frac{1}{m} \Sigma_{i=1}^m 1\{z^{(i)} = j\},\\ \mu_j &= \frac{\Sigma_{i=1}^m 1\{z^{(i)} = j\}x^{(i)}}{\Sigma_{i=1}^m 1\{z^{(i)} = j\}},\\ \Sigma_j &= \frac{\Sigma_{i=1}^m 1\{z^{(i)} = j\}(x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\Sigma_{i=1}^m 1\{z^{(i)} = j\}}.\end{aligned}$$

But in reality, we don’t have the value of z, all we have is unlabeled dataset. We have to **guess** it.

### EM Algorithm

Repeat until convergence {

​	E-Step (Expectation, guess value of the unknown $z^{(i)}$’s):

​	Set:

 $$\begin{aligned}w^{(i)}_j &\gets P(z^{(i)} = j | x^{(i)}; \phi,\mu, \Sigma)\\ &= \frac{P(x^{(i)}|z^{(i)} = j) P(z^{(i)} = j)}{\Sigma_{l=1}^kP(x^{(i)}|z^{(i)} = l)P(z^{(i)} = l)}\\ &= \frac{\frac{1}{\sqrt{2\pi} \cdot |\Sigma_j|^{1/2}}exp\left[(x^{(i)} - \mu_j)^T\Sigma_j^{-1}(x^{(i)} - \mu_j)\right]\cdot \phi_j}{\Sigma_{l=1}^k\frac{1}{\sqrt{2\pi} \cdot |\Sigma_l|^{1/2}}exp\left[(x^{(i)} - \mu_l)^T\Sigma_l^{-1}(x^{(i)} - \mu_l)\right]\cdot \phi_l}\end{aligned}$$



​	M-Step (Maximization):

​	$$\begin{aligned}\phi_j &\gets \frac{1}{m} \Sigma_{i=1}^m w_j^{(i)}\\ \mu_j &\gets \frac{\Sigma_{i=1}^m w_j^{(i)} x^{(i)}}{\Sigma_{i=1}^m w_j^{(i)}}\\ \Sigma_j &\gets \frac{\Sigma_{i=1}^m w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\Sigma_{i=1}^m w_j^{(i)}}\end{aligned}$$

}

In GDA, where we have the class labels, the ML Estimation:

$\phi_j = P(z^{(i)} = j) = \frac{1}{m} \Sigma_{i=1}^m \{z^{(i)} = j\}$

In EM, we only have $x^{(i)}$, but the class labels $z^{(i)}$ are unknown, we have the guess for the value of $z$ (E-Step): $w_j^{(i)} = P(z^{(i)} = j | x^{(i)})$

### Jensen’s Inequality

Let $f$ be a convex function (e.g., $f^{‘’}(x) \geq 0$), let $X$ be a random variable, then,

$$f(EX) \leq E[f(x)]$$

Further, if $f^{‘’}(x) > 0$ ($f$ is strictly convex), then $E[f(x)] = f(EX) \iff X = E[X]$, i.e., if $X$ is a constant.

For concave case, everything holds for the inequality signs reversed.

Here we have some model for $P(x,z;\theta)$, we observe any $x$, maximize:

$$\begin{aligned}l(\theta) &= \Sigma_{i=1}^m \log P(x^{(i)};\theta)\\ &= \Sigma_{i=1}^m \log \Sigma_{z^{(i)}} P(x^{(i)}, z^{(i)};\theta) \end{aligned}$$

We have,

$$\begin{aligned}&\underset{\theta}{max} \Sigma_i \log P(x^{(i)};\theta)\\&=\underset{\theta}{max} \log \Sigma_{z^{(i)}} P(x^{(i)}, z^{(i)};\theta)\\ &= \underset{\theta}{max} \Sigma_i \log \Sigma_{z^{(i)}} Q_i(z^{(i)}) \frac{P(x^{(i)}, z^{(i)};\theta)}{Q_i(z^{(i)})}\end{aligned}$$

Where $Q_i(z^{(i)}) \geq 0, \Sigma_{z^{(i)}} Q_i(z^{(i)}) = 1$,

we then have the above formula can be derived as,

$= \underset{\theta}{max} \Sigma_i \log E_{z^{(i)} \sim Q_i} \left[\frac{P(x^{(i)}, z^{(i)};\theta)}{Q(z^{(i)})}\right]$

Since the $\log$ function is concave, we have, from Jensen’s Inequality, $\log E[X] \geq E[\log X]$,

We then have the above result,

$\begin{aligned}\Sigma_i \log \mathbb{E}_{z^{(i)} \sim Q_i} \left[\frac{P(x^{(i)}, z^{(i)};\theta)}{Q_i(z^{(i)})}\right] &\geq \Sigma_i \mathbb{E}_{z^{(i)}}\left[\log \frac{P(x^{(i)}, z^{(i)};\theta)}{Q_i(z^{(i)})}\right]\\ &= \Sigma_i \Sigma_{z^{(i)}} Q_i(z^{(i)}) \log \frac{P(x^{(i)}, z^{(i)};\theta)}{Q_i(z^{(i)})}\end{aligned}$

We’ve just shown that the log-likelihood $l(\theta)$ has a lower bound,

$$l(\theta) \geq \Sigma_i\Sigma_{z^{(i)}} \log \frac{P(x^{(i)}, z^{(i)};\theta)}{Q_i(z^{(i)})}$$

To ensure the lower bound is tight enough (goes to equality), we need the inequality in Jensen’s equality to go to equality, i.e.,

$$\frac{P(x^{(i)}, z^{(i)};\theta)}{Q_i(z^{(i)})} = constant$$, w.r.t. $\forall z^{(i)}$,

we have $Q_i(z^{(i)}) \propto P(x^{(i)}, z^{(i)};\theta), \Sigma_{z^{(i)}} Q_i(z^{(i)}) = 1$, thus the choice of $Q_i(z^{(i)})$ would be,

$\begin{aligned}Q(z^{(i)}) &= \frac{P(x^{(i)}, z^{(i)};\theta)}{\Sigma_{z^{(i)}}P(x^{(i)}, z^{(i)};\theta)}\\ &= \frac{P(x^{(i)}, z^{(i)};\theta)}{P(x^{(i)};\theta)}\\ &= P(z^{(i)}|x^{(i)};\theta)\end{aligned}$

Therefore, the EM algorithm has two steps:

1. E-Step:

   Set $Q_i(z^{(i)}) = P(z^{(i)}|x^{(i)} ;\theta)$, which by this time we create a lower bound for the log-likelihood of $\theta$

2. M-Step:

   $\Theta \gets arg \underset{\Theta}{max} \Sigma_i \Sigma_{z^{(i)}} Q_i(z^{(i)}) \log \frac{P(x^{(i)},z^{(i)};\Theta)}{Q_i(z^{(i)})}$

   






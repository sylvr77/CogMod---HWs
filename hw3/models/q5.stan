data {
    int<lower=1> N;
    vector[N] x;
    vector[N] y; 
}

parameters {
    real alpha;
    real beta;
    real<lower=0> sigma2;
}

transformed parameters {
    real<lower=0> sigma;
    sigma = sqrt(sigma2);
}

model {
    // Prior
    target += inv_gamma_lpdf(sigma2 | 1, 1);
    target += normal_lpdf(alpha | 0, 10);
    target += normal_lpdf(beta | 0, 10);

    // Vectorized sampling statement
    target += normal_lpdf(y | alpha + beta * x, sigma);
}

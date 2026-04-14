data {
    int<lower=1> N;
    array[N] int<lower=0,upper=1> y;
    matrix[N, 3] X;
}

parameters {
    real beta_0;

    // for the 3 chosen attrs...
    vector[3] beta;
}

model {
    // Priors
    beta_0 ~ normal(0, 2.5);
    beta ~ normal(0, 2.5);

    // Likelihood
    y ~ bernoulli_logit(beta_0 + X*beta);
}

/*
// with our uncovered parameter betas, we try to predict output...
generated quantities {
    vector[N] p_hat;
    array[N] int y_rep;

    for (n in 1:N) {
        p_hat[n] = inv_logit(beta_0 + X[n] * beta);
        y_rep[n] = bernoulli_rng(p_hat[n]);
    }
}
*/
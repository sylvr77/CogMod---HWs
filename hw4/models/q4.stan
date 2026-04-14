data {
    // train set...
    int<lower=1> N_train;
    array[N_train] int<lower=0,upper=1> y_train;
    matrix[N_train, 3] X_train;

    // test set...
    int<lower=1> N_test;
    matrix[N_test, 3] X_test;
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
    y_train ~ bernoulli_logit(beta_0 + X_train*beta);
}

// with our uncovered parameter betas, we try to predict output...
generated quantities {
    vector[N_test] p_hat;

    for (n in 1:N_test) {
        p_hat[n] = inv_logit(beta_0 + X_test[n]*beta);
    }
}
data {
    int<lower=1> num_feats;

    // train set...
    int<lower=1> N_train;
    array[N_train] int<lower=0,upper=1> y_train;
    matrix[N_train, num_feats] X_train;

    // test set...
    int<lower=1> N_test;
    matrix[N_test, num_feats] X_test;
}

parameters {
    real beta_0;

    vector[num_feats] beta;
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
    // calculated probs for the predictive model accuracy...
    vector[N_test] p_hat;

    // vectorised computation for efficiency...
    p_hat = inv_logit(beta_0 + X_test*beta);


    // simulated data for the generative model accuracy...
    array[N_test] int y_sim;

    for (n in 1:N_test) {
        y_sim[n] = bernoulli_logit_rng(beta_0 + X_test[n]*beta);
    }
}
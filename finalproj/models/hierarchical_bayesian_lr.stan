data {
    int<lower=1> num_feats;
    int<lower=1> num_unique;

    // train set...
    int<lower=1> N_train;
    array[N_train] int<lower=0,upper=1> y_train;
    matrix[N_train, num_feats] X_train;
    array[N_train] int<lower=1,upper=num_unique> person_train;

    // test set...
    int<lower=1> N_test;
    matrix[N_test, num_feats] X_test;
    array[N_test] int<lower=1,upper=num_unique> person_test;

}

parameters {
    real mu_alpha;
    real<lower=0> sigma_alpha;
    vector[num_unique] alpha_raw;
    vector[num_feats] beta;
}

transformed parameters{
    vector[num_unique] alpha;

    alpha = mu_alpha + sigma_alpha*alpha_raw;
}

model {
    // Priors
    mu_alpha ~ normal(0,2.5);
    sigma_alpha ~ exponential(1);
    alpha_raw ~ normal(0,1);
    beta ~ normal(0, 2.5);

    // Likelihood
    y_train ~ bernoulli_logit(alpha[person_train] + X_train*beta);
}

generated quantities {
    // calculated probs for the predictive model accuracy...
    vector[N_test] p_hat;



    // simulated data for the generative model accuracy...
    array[N_test] int y_sim;

    for (n in 1:N_test) {
        p_hat[n] = inv_logit(alpha[person_test[n]] + X_test[n]*beta);

        y_sim[n] = bernoulli_logit_rng(alpha[person_test[n]] + X_test[n]*beta);
    }
}
data {
    int<lower=1> N_old; 
    int<lower=1> N_new; 
    array[2] int<lower=0> y_old;
    array[2] int<lower=0> y_new;
}

parameters {
    real<lower=0, upper=1> d;
    real<lower=0, upper=1> g;
}

transformed parameters {
    // each tree needs to calculate both the old and new conditional probabilities...
    simplex[2] old_tree;
    simplex[2] new_tree;

    // as taken from slide12, and described in the ipynb file,
    //   we now use the 2HT probabilities...
    // p(old | old)
    old_tree[1] = d + (1 - d)*g;
    // p(new | old)
    old_tree[2] = (1 - d)*(1 - g);
    // p(old | new)
    new_tree[1] = (1 - d)*g;
    // p(new | new)
    new_tree[2] = d + (1 - d)*(1 - g);
}

model {
    d ~ beta(1,1);
    g ~ beta(1,1);

    // Likelihood
    y_old ~ multinomial(old_tree);
    y_new ~ multinomial(new_tree);
}

generated quantities{
    array[2] int pred_y_old;
    array[2] int pred_y_new;
    real log_likelihood;

    pred_y_old = multinomial_rng(old_tree, N_old);
    pred_y_new = multinomial_rng(new_tree, N_new);

    log_likelihood = multinomial_lpmf(y_old | old_tree) 
    + multinomial_lpmf(y_new | new_tree);
}
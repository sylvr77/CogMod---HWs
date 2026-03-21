data {
    int<lower=1> N; // number of rows
    int<lower=1> J; // number of people
    array[N] int<lower=1, upper=J> id;   // person ID for each trial/row
    array[N] real<lower=0> y; // response time
    array[N] int<lower=1, upper=2> condition; // condition indicator
    array[N] int<lower=0, upper=1> choice; // choice indicator
}

transformed data {
  // init a vector to hold the minimum response times
  real eps = 1e-6;
  array[J] real rt_min;

  // initialize to +inf
  for (j in 1:J) {
    rt_min[j] = positive_infinity();
  }

  // compute per-person minimum RT
  for (n in 1:N) {
    if (y[n] < rt_min[id[n]]) {
        rt_min[id[n]] = y[n];
    }
  }
}

parameters {
    // Each person has their own parameters
    // Your code here: One of these parameters should index the
    // difficulty of the task. Which one? You will then have to
    // assume two parameters vectors, one for each condition. :)


    // from the description of the problem, the main differentiator
    // between each interference field is the difficulty of the sensor
    // signatures, which is the evidence for this problem.
    // 
    // the parameter that best reflects task difficulty is v, drift rate.
    // tau is a constant that relates to hitting time, beta controls the 
    // starting threshold, and a controls how much evidence is needed, but 
    // not relates to the type of evidence itself.
    // 
    // drift rate on the other hand talks about how reliable the evidence is, 
    // which matches the differentiator in this problem the closet out of all
    // the parameters.


    array[J] real v1;
    array[J] real v2;

    array[J] real<lower=0> a;
    array[J] real<lower=0, upper=1> beta;
    array[J] real<lower=0, upper=1> tau_raw;
}

transformed parameters {
  // A good way to bound non-decision time (tau)
  array[J] real<lower=0> tau;
  for (j in 1:J) {
    tau[j] = tau_raw[j] * (rt_min[j] - eps);
  }
}

model {
    // Priors
    for (j in 1:J){
        v1[j] ~ normal(0, 2);
        v2[j] ~ normal(0, 2);

        a[j] ~ normal(1, 0.5);
        beta[j] ~ beta(2, 2);
        tau_raw[j] ~ beta(2, 2);
    }
    
    
    // Likelihood
    for (n in 1:N) {
        // Condition 1
        if (condition[n] == 1) {
            if (choice[n] == 1) {
                // Your code here 
                // Hint: use something like param[id[n]] 
                // to index the person-specific parameters ;)

                target += wiener_lpdf(y[n] | a[id[n]], tau[id[n]], beta[id[n]], v1[id[n]]);
            } 

            else {
                target += wiener_lpdf(y[n] | a[id[n]], tau[id[n]], 1 - beta[id[n]], -v1[id[n]]);
            }
        }

        // Condition 2
        if (condition[n] == 2) {
            if (choice[n] == 1) {
                target += wiener_lpdf(y[n] | a[id[n]], tau[id[n]], beta[id[n]], v2[id[n]]);
            } 
            
            else {
                target += wiener_lpdf(y[n] | a[id[n]], tau[id[n]], 1 - beta[id[n]], -v2[id[n]]);
            }
        }
    }
}
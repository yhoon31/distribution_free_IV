library(randomForest)
library(nnet)
library(parallel)
library(pbapply)

p <- 10
K <- 10
theta <- matrix(seq(-1.5, 1.5, length.out = K * p), nrow = K, ncol = p)
mu_coef <- c(1, 0.8, 0.6, 0.4, 0.2, -0.2, -0.4, -0.6, -0.8, -1)
Gamma <- 0.8

make_X_df <- function(X) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  X_df <- as.data.frame(X, stringsAsFactors = FALSE)
  colnames(X_df) <- paste0("X", seq_len(ncol(X_df)))
  X_df
}

generate_data <- function(n, Gamma) {
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta <- 0.5 * X %*% t(theta)
  e <- exp(eta - apply(eta, 1, max))
  e <- e / rowSums(e)
  Z <- apply(e, 1, function(prob) sample.int(K, 1, prob = prob))
  w <- e / apply(e, 1, max)
  pi_z <- Gamma * w[cbind(seq_len(n), Z)]
  mu <- as.vector(X %*% mu_coef)
  B <- sample(c(-1, 1), n, replace = TRUE)
  Y <- mu + 2.5 * B + 0.25 * rnorm(n)
  h <- ifelse(Y > mu, 1, -1)
  pR <- pi_z + 0.8 * (pi_z / Gamma) * (Gamma - pi_z) * h
  R <- rbinom(n, 1, pR)
  list(X = X, Z = Z, R = R, Y = Y)
}

n_train <- 2000
dat_train <- generate_data(n_train, Gamma)
X_train <- as.matrix(dat_train$X)
X_train_df <- make_X_df(X_train)
idx_obs <- dat_train$R == 1

fit_mu <- randomForest(x = X_train[idx_obs, , drop = FALSE], y = dat_train$Y[idx_obs], ntree = 500, nodesize = 5)
predict_mu_hat <- function(X) as.numeric(predict(fit_mu, newdata = as.matrix(X)))

fit_e <- multinom(Z ~ ., data = data.frame(X_train_df, Z = factor(dat_train$Z)), trace = FALSE, maxit = 300, MaxNWts = 10000)

fit_pi <- vector("list", K)
for (z in seq_len(K)) {
  idx_z <- dat_train$Z == z
  fit_pi[[z]] <- randomForest(x = X_train[idx_z, , drop = FALSE], y = factor(dat_train$R[idx_z]), ntree = 300, nodesize = 10)
}

fit_f_mean <- vector("list", K)
fit_f_sq <- vector("list", K)
for (z in seq_len(K)) {
  idx_z <- dat_train$Z == z & dat_train$R == 1
  fit_f_mean[[z]] <- randomForest(x = X_train[idx_z, , drop = FALSE], y = dat_train$Y[idx_z], ntree = 300, nodesize = 10)
  fit_f_sq[[z]] <- randomForest(x = X_train[idx_z, , drop = FALSE], y = dat_train$Y[idx_z]^2, ntree = 300, nodesize = 10)
}

predict_nuisance <- function(X) {
  X <- as.matrix(X)
  n <- nrow(X)
  e_hat <- as.matrix(predict(fit_e, newdata = make_X_df(X), type = "probs"))
  if (ncol(e_hat) != K) {
    tmp <- matrix(0, nrow = n, ncol = K)
    tmp[, seq_len(ncol(e_hat))] <- e_hat
    e_hat <- tmp
  }
  e_hat <- pmax(e_hat, 1e-8)
  e_hat <- e_hat / rowSums(e_hat)
  
  pi_hat <- matrix(0, nrow = n, ncol = K)
  for (z in seq_len(K)) {
    pr <- predict(fit_pi[[z]], newdata = X, type = "prob")
    pi_hat[, z] <- if ("1" %in% colnames(pr)) pr[, "1"] else pr[, ncol(pr)]
  }
  pi_hat <- pmin(pmax(pi_hat, 1e-6), 1)
  
  mu_f <- matrix(0, nrow = n, ncol = K)
  var_f <- matrix(0, nrow = n, ncol = K)
  for (z in seq_len(K)) {
    mu_f[, z] <- predict(fit_f_mean[[z]], newdata = X)
    EY2 <- predict(fit_f_sq[[z]], newdata = X)
    var_f[, z] <- pmax(EY2 - mu_f[, z]^2, 0.05^2)
  }
  list(e_hat = e_hat, pi_hat = pi_hat, mu_f = mu_f, sd_f = sqrt(var_f))
}

true_nuisance <- function(X, Gamma) {
  X <- as.matrix(X)
  eta <- 0.5 * X %*% t(theta)
  e <- exp(eta - apply(eta, 1, max))
  e <- e / rowSums(e)
  w <- e / apply(e, 1, max)
  pi <- Gamma * w
  mu <- as.vector(X %*% mu_coef)
  list(e = e, pi = pi, mu = mu, w = w)
}

true_y_density_vec <- function(y, mu) 0.5 * dnorm(y, mean = mu - 2.5, sd = 0.25) + 0.5 * dnorm(y, mean = mu + 2.5, sd = 0.25)

true_f_density_vec <- function(y, mu, pi, Gamma) {
  h <- ifelse(y > mu, 1, -1)
  true_y_density_vec(y, mu) * (1 + 0.8 * (Gamma - pi) / Gamma * h)
}

true_q_density_vec <- function(y, mu, pi, Gamma) {
  h <- ifelse(y > mu, 1, -1)
  true_y_density_vec(y, mu) * (pi + 0.8 * (pi / Gamma) * (Gamma - pi) * h)
}

nuisance_train <- predict_nuisance(X_train)
qhat_train <- matrix(0,nrow=sum(idx_obs),ncol=K)
for(z in seq_len(K)) qhat_train[,z] <- nuisance_train$pi_hat[idx_obs,z]*dnorm(dat_train$Y[idx_obs],mean=nuisance_train$mu_f[idx_obs,z],sd=nuisance_train$sd_f[idx_obs,z])

policy_label <- max.col(qhat_train,ties.method="first")
policy_levels <- sort(unique(policy_label))
if(length(policy_levels)==1) policy_fit <- list(constant=policy_levels[1]) else policy_fit <- randomForest(x=data.frame(make_X_df(X_train[idx_obs,,drop=FALSE]),Y=dat_train$Y[idx_obs]),y=factor(policy_label,levels=policy_levels),ntree=300,nodesize=10)
predict_policy <- function(policy_fit,X,Y){X <- as.matrix(X); if(!is.null(policy_fit$constant)) rep(policy_fit$constant,nrow(X)) else as.integer(as.character(predict(policy_fit,newdata=data.frame(make_X_df(X),Y=Y))))}



compute_delta_true_fast <- function(X, muhat, t_grid, Gamma) {
  mu <- as.vector(as.matrix(X) %*% mu_coef)
  n <- length(mu)
  out <- matrix(0, nrow = n, ncol = length(t_grid))
  for (j in seq_along(t_grid)) {
    lo <- muhat - t_grid[j]
    hi <- muhat + t_grid[j]
    prob <- 0.5 * (pnorm(hi, mu - 2.5, 0.25) - pnorm(lo, mu - 2.5, 0.25)) + 0.5 * (pnorm(hi, mu + 2.5, 0.25) - pnorm(lo, mu + 2.5, 0.25))
    out[, j] <- 1 - Gamma * prob
  }
  out
}

compute_delta_plugin_fast <- function(X, muhat, t_grid, nuisance, n_y = 601) {
  n <- nrow(X)
  out <- matrix(0, nrow = n, ncol = length(t_grid))
  tmax <- max(t_grid)
  for (i in seq_len(n)) {
    y_grid <- seq(muhat[i] - tmax, muhat[i] + tmax, length.out = n_y)
    Q <- sapply(seq_len(K), function(z) nuisance$pi_hat[i, z] * dnorm(y_grid, nuisance$mu_f[i, z], nuisance$sd_f[i, z]))
    qmax <- apply(Q, 1, max)
    dy <- y_grid[2] - y_grid[1]
    cumint <- c(0, cumsum((qmax[-length(qmax)] + qmax[-1]) * 0.5 * dy))
    F_hi <- approx(y_grid, cumint, xout = muhat[i] + t_grid, rule = 2)$y
    F_lo <- approx(y_grid, cumint, xout = muhat[i] - t_grid, rule = 2)$y
    out[i, ] <- 1 - (F_hi - F_lo)
  }
  pmin(pmax(out, 0), 1)
}

draw_true_f <- function(mu, pi, Gamma) {
  n <- length(mu)
  Y <- numeric(n)
  accepted <- rep(FALSE, n)
  a <- 0.8 * (Gamma - pi) / Gamma
  M <- 1 + a
  while (any(!accepted)) {
    idx <- which(!accepted)
    B <- sample(c(-1, 1), length(idx), replace = TRUE)
    y_prop <- mu[idx] + 2.5 * B + 0.25 * rnorm(length(idx))
    h <- ifelse(y_prop > mu[idx], 1, -1)
    acc <- runif(length(idx)) <= (1 + a[idx] * h) / M[idx]
    Y[idx[acc]] <- y_prop[acc]
    accepted[idx[acc]] <- TRUE
  }
  Y
}

generate_true_mc <- function(X, muhat, Gamma, n_mc) {
  n <- nrow(X)
  nui <- true_nuisance(X, Gamma)
  id_i <- rep(seq_len(n), each = K * n_mc)
  id_z <- rep(rep(seq_len(K), each = n_mc), times = n)
  pi_vec <- nui$pi[cbind(id_i, id_z)]
  Y_mc <- draw_true_f(nui$mu[id_i], pi_vec, Gamma)
  list(id_i = id_i, id_z = id_z, Y = Y_mc, S = abs(Y_mc - muhat[id_i]))
}

generate_estimated_mc <- function(X, muhat, nuisance, n_mc) {
  n <- nrow(X)
  id_i <- rep(seq_len(n), each = K * n_mc)
  id_z <- rep(rep(seq_len(K), each = n_mc), times = n)
  mu_vec <- nuisance$mu_f[cbind(id_i, id_z)]
  sd_vec <- nuisance$sd_f[cbind(id_i, id_z)]
  Y_mc <- mu_vec + sd_vec * rnorm(length(id_i))
  list(id_i = id_i, id_z = id_z, Y = Y_mc, S = abs(Y_mc - muhat[id_i]))
}

compute_M_fast <- function(A_mc, S_mc, id_i, id_z, pi_mat, n, t_grid, n_mc) {
  M <- array(0, dim = c(n, K, length(t_grid)))
  for (z in seq_len(K)) {
    idx_z <- which(id_z == z)
    A_z <- A_mc[idx_z]
    S_z <- S_mc[idx_z]
    for (i in seq_len(n)) {
      idx <- ((i - 1) * n_mc + 1):(i * n_mc)
      scores <- sort(S_z[idx][A_z[idx] == z])
      if (length(scores) > 0) M[i, z, ] <- pi_mat[i, z] * findInterval(t_grid, scores) / n_mc
    }
  }
  M
}

compute_oracle_eif_fast <- function(dat, t_grid, true_nui, muhat, Gamma) {
  n <- nrow(dat$X)
  zstar <- max.col(true_nui$pi, ties.method = "first")
  mu <- true_nui$mu
  Mstar <- matrix(0, nrow = n, ncol = length(t_grid))
  
  for (j in seq_along(t_grid)) {
    lo <- muhat - t_grid[j]
    hi <- muhat + t_grid[j]
    prob <- 0.5 * (pnorm(hi, mu - 2.5, 0.25) - pnorm(lo, mu - 2.5, 0.25)) + 0.5 * (pnorm(hi, mu + 2.5, 0.25) - pnorm(lo, mu + 2.5, 0.25))
    Mstar[, j] <- Gamma * prob
  }
  
  S_obs <- abs(dat$Y - muhat)
  e_obs <- true_nui$e[cbind(seq_len(n), dat$Z)]
  EIF <- 1 - Mstar
  
  for (i in seq_len(n)) {
    H_i <- dat$R[i] * (S_obs[i] <= t_grid) * (dat$Z[i] == zstar[i])
    M_obs <- if (dat$Z[i] == zstar[i]) Mstar[i, ] else rep(0, length(t_grid))
    EIF[i, ] <- EIF[i, ] - (H_i - M_obs) / e_obs[i]
  }
  colMeans(EIF)
}

compute_eif_curve <- function(dat, t_grid, M, A_obs, e_mat, muhat) {
  n <- nrow(dat$X)
  z <- dat$Z
  e_obs <- e_mat[cbind(seq_len(n), z)]
  S_obs <- abs(dat$Y - muhat)
  M_sum <- apply(M, c(1, 3), sum)
  EIF <- 1 - M_sum
  for (i in seq_len(n)) {
    H_i <- dat$R[i] * (S_obs[i] <= t_grid) * (A_obs[i] == z[i])
    EIF[i, ] <- EIF[i, ] - (H_i - M[i, z[i], ]) / e_obs[i]
  }
  colMeans(EIF)
}


get_t_conformal <- function(delta_matrix,alpha,t_grid){criterion <- (colSums(delta_matrix)+1)/(nrow(delta_matrix)+1); valid <- which(criterion<=alpha); if(length(valid)==0) return(Inf); t_grid[min(valid)]}
get_t_eif <- function(eif_curve,alpha,t_grid){
  valid <- which(eif_curve >= alpha)
  if(length(valid)==0) return(-Inf)
  if(max(valid)==length(t_grid)) return(Inf)
  t_grid[max(valid)]
}

tn <- 200
n_cal <- 500
n_test <- 500
alpha_set <- seq(0.05, 0.30, by = 0.05)
t_grid <- seq(0, 6, by = 0.02)
n_mc <- 50
n_y <- 601
n_cores <- max(1, detectCores() - 1)
method_names <- c("true_oracle", "plugin", "eif_oracle_policy", "eif_learned_policy_true", "eif_estimated")

cl <- makeCluster(n_cores)
clusterEvalQ(cl, {
  library(randomForest)
  library(nnet)
})

clusterExport(cl, c("p", "K", "theta", "mu_coef", "Gamma", "make_X_df", "generate_data", "fit_mu", "fit_e", "fit_pi", "fit_f_mean", "fit_f_sq", "predict_mu_hat", "predict_nuisance", "true_nuisance", "policy_fit", "predict_policy", "compute_delta_true_fast", "compute_delta_plugin_fast", "draw_true_f", "generate_true_mc", "generate_estimated_mc", "compute_M_fast", "compute_oracle_eif_fast", "compute_eif_curve", "get_t_conformal", "get_t_eif", "n_cal", "n_test", "alpha_set", "t_grid", "n_mc", "n_y", "method_names"), envir = .GlobalEnv)

results_list <- pblapply(seq_len(tn), function(trial) {
  dat_cal <- generate_data(n_cal, Gamma)
  X_cal <- as.matrix(dat_cal$X)
  muhat_cal <- predict_mu_hat(X_cal)
  nuisance_cal <- predict_nuisance(X_cal)
  true_cal <- true_nuisance(X_cal, Gamma)
  
  X_test <- matrix(rnorm(n_test * p), nrow = n_test, ncol = p)
  B_test <- sample(c(-1, 1), n_test, replace = TRUE)
  Y_test <- as.vector(X_test %*% mu_coef) + 2.5 * B_test + 0.25 * rnorm(n_test)
  muhat_test <- predict_mu_hat(X_test)
  S_test <- abs(Y_test - muhat_test)
  
  delta_true <- compute_delta_true_fast(X_cal, muhat_cal, t_grid, Gamma)
  delta_plugin <- compute_delta_plugin_fast(X_cal, muhat_cal, t_grid, nuisance_cal, n_y)
  eif_oracle_policy <- compute_oracle_eif_fast(dat_cal, t_grid, true_cal, muhat_cal, Gamma)
  
  mc_true <- generate_true_mc(X_cal, muhat_cal, Gamma, n_mc)
  X_mc_true <- X_cal[mc_true$id_i, , drop = FALSE]
  A_mc_learned_true <- predict_policy(policy_fit, X_mc_true, mc_true$Y)
  M_learned_true <- compute_M_fast(A_mc_learned_true, mc_true$S, mc_true$id_i, mc_true$id_z, true_cal$pi, n_cal, t_grid, n_mc)
  A_obs_learned <- predict_policy(policy_fit, X_cal, dat_cal$Y)
  eif_learned_policy_true <- compute_eif_curve(dat_cal, t_grid, M_learned_true, A_obs_learned, true_cal$e, muhat_cal)
  
  mc_est <- generate_estimated_mc(X_cal, muhat_cal, nuisance_cal, n_mc)
  X_mc_est <- X_cal[mc_est$id_i, , drop = FALSE]
  A_mc_est <- predict_policy(policy_fit, X_mc_est, mc_est$Y)
  M_est <- compute_M_fast(A_mc_est, mc_est$S, mc_est$id_i, mc_est$id_z, nuisance_cal$pi_hat, n_cal, t_grid, n_mc)
  eif_estimated <- compute_eif_curve(dat_cal, t_grid, M_est, A_obs_learned, nuisance_cal$e_hat, muhat_cal)
  
  result <- numeric(2 * length(method_names) * length(alpha_set))
  for (j in seq_along(alpha_set)) {
    alpha <- alpha_set[j]
    t_values <- c(get_t_conformal(delta_true, alpha, t_grid), get_t_conformal(delta_plugin, alpha, t_grid), get_t_eif(eif_oracle_policy, alpha, t_grid), get_t_eif(eif_learned_policy_true, alpha, t_grid), get_t_eif(eif_estimated, alpha, t_grid))
    for (m in seq_along(method_names)) {
      idx <- (j - 1) * 2 * length(method_names) + 2 * m - 1
      result[idx] <- mean(S_test <= t_values[m])
      result[idx + 1] <- 2 * t_values[m]
    }
  }
  result
}, cl = cl)

stopCluster(cl)

results <- do.call(rbind, results_list)
result_names <- unlist(lapply(alpha_set, function(alpha) unlist(lapply(method_names, function(method) c(paste0(method, "_coverage_alpha_", alpha), paste0(method, "_width_alpha_", alpha))))))
colnames(results) <- result_names

summary_list <- lapply(method_names, function(method) {
  coverage <- results[, grep(paste0("^", method, "_coverage"), colnames(results)), drop = FALSE]
  width <- results[, grep(paste0("^", method, "_width"), colnames(results)), drop = FALSE]
  list(coverage_mean = colMeans(coverage), coverage_se = apply(coverage, 2, sd) / sqrt(tn), width_mean = colMeans(width), width_se = apply(width, 2, sd) / sqrt(tn))
})
names(summary_list) <- method_names

res <- do.call(cbind, lapply(method_names, function(method) {
  x <- summary_list[[method]]
  cbind(x$coverage_mean, x$coverage_se, x$width_mean, x$width_se)
}))
colnames(res) <- unlist(lapply(method_names, function(method) c(paste0(method, "_coverage_mean"), paste0(method, "_coverage_se"), paste0(method, "_width_mean"), paste0(method, "_width_se"))))
rownames(res) <- paste0("alpha_", alpha_set)
print(res)

#write.table(res, paste0("policy_Gamma_", gsub("\\.", "", as.character(Gamma)), "_setting_2_tr_", n_train, "_new.txt"))

plot_methods <- c("true_oracle", "plugin", "eif_oracle_policy", "eif_estimated")
coverage_mean <- sapply(plot_methods,function(method) summary_list[[method]]$coverage_mean)

plot_names <- c("Semi-oracle", "Plug-in", "EIF Oracle", "Policy")
plot_cols <- c("#D95F5F", "#55A868", "#9467BD", "#4F81BD")
plot_pch <- c(16, 17, 15, 19)

par(mfrow = c(1, 1), mgp = c(3.5, 1, 0), cex.lab = 2, cex.axis = 1.7, mar = c(2.5, 5, 4, 0), oma = c(2.5, 1, 0, 2), xpd = NA, font.main = 1, cex.main = 2)

matplot(1 - alpha_set, coverage_mean, type = "p", pch = plot_pch, col = plot_cols, xlab = expression(1 - alpha), ylab = "Coverage rate", lwd = 2, ylim = c(0.5, 1), main = "")
for (m in seq_along(plot_methods)) arrows(1 - alpha_set, coverage_mean[, m] - coverage_se[, m], 1 - alpha_set, pmin(coverage_mean[, m] + coverage_se[, m], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = plot_cols[m])
segments(min(1 - alpha_set), min(1 - alpha_set), max(1 - alpha_set), max(1 - alpha_set), lty = 2, lwd = 2)

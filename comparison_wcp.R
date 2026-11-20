library(randomForest)
library(MASS)
library(parallel)
library(pbapply)

p <- 5
K <- 10
n_train <- 2000
n_cal <- 500
n_test <- 500
tn <- 200
alpha_set <- seq(0.05, 0.30, by = 0.05)
t_grid <- seq(0, 8, by = 0.1)
n_mc <- 50
n_cores <- max(1, detectCores() - 1)

Sigma <- outer(1:p, 1:p, function(i, j) 0.5^abs(i - j))
z_base <- seq(0.4, -0.4, length.out = K)
z_x1 <- seq(-1, 1, length.out = K)
z_x2 <- seq(-0.2, 0.2, length.out = K)
z_x1_3 <- seq(-0.1, 0.1, length.out = K)
z_x2_3 <- seq(-0.02, 0.02, length.out = K)

mu_fun <- function(X) {
  X[, 1] + 0.8 * sin(X[, 2]) + 0.5 * X[, 3]^2 + 0.5 * X[, 1] * X[, 2]
}

prob_z_fun <- function(X) {
  eta <- outer(X[, 1], z_x1) + outer(X[, 2], z_x2)
  eta <- sweep(eta, 2, z_base, "+")
  eta <- eta - apply(eta, 1, max)
  pz <- exp(eta)
  pz / rowSums(pz)
}

prob_z_fun_3 <- function(X) {
  eta <- outer(X[, 1], z_x1_3) + outer(X[, 2], z_x2_3)
  eta <- sweep(eta, 2, z_base, "+")
  eta <- eta - apply(eta, 1, max)
  pz <- exp(eta)
  pz / rowSums(pz)
}

generate_data_1 <- function(n) {
  X <- MASS::mvrnorm(n, rep(0, p), Sigma)
  pz <- prob_z_fun(X)
  Z <- apply(pz, 1, function(pr) sample.int(K, 1, prob = pr)) - 1
  mu <- mu_fun(X)
  eps <- rnorm(n)
  Y <- mu + eps
  p_r <- ifelse(Z == 1, 0.995 - 0.005 * exp(-eps^2), 0.4 * exp(-eps^2))
  R <- rbinom(n, 1, p_r)
  list(X = X, Z = Z, Y = Y, R = R, mu = mu)
}

generate_data_2 <- function(n) {
  X <- MASS::mvrnorm(n, rep(0, p), Sigma)
  pz <- prob_z_fun(X)
  Z <- apply(pz, 1, function(pr) sample.int(K, 1, prob = pr)) - 1
  mu <- mu_fun(X)
  eps <- rnorm(n)
  Y <- mu + eps
  p_r <- ifelse(Z == 1, 0.9 - 0.03 * exp(-eps^2), 0.4 * exp(-eps^2))
  R <- rbinom(n, 1, p_r)
  list(X = X, Z = Z, Y = Y, R = R, mu = mu)
}

generate_data_3 <- function(n) {
  X <- MASS::mvrnorm(n, rep(0, p), Sigma)
  pz <- prob_z_fun_3(X)
  Z <- apply(pz, 1, function(pr) sample.int(K, 1, prob = pr)) - 1
  mu <- mu_fun(X)
  eps <- rnorm(n)
  Y <- mu + eps
  p_r <- ifelse(Z == 1, 0.90 - 0.03 * exp(-eps^2), 0.80 + 0.03 * exp(-eps^2))
  R <- rbinom(n, 1, p_r)
  list(X = X, Z = Z, Y = Y, R = R, mu = mu)
}

generate_data <- function(n, scenario) {
  if (scenario == 1) return(generate_data_1(n))
  if (scenario == 2) return(generate_data_2(n))
  if (scenario == 3) return(generate_data_3(n))
  stop("scenario must be 1, 2, or 3")
}

prob_z <- function(X, scenario) {
  if (scenario == 3) prob_z_fun_3(X) else prob_z_fun(X)
}

true_pi <- function(Z, scenario) {
  ez <- 1 / sqrt(3)
  if (scenario == 1) {
    pi_high <- 0.995 - 0.005 * ez
    pi_low <- 0.4 * ez
  } else if (scenario == 2) {
    pi_high <- 0.9 - 0.03 * ez
    pi_low <- 0.4 * ez
  } else {
    pi_high <- 0.90 - 0.03 * ez
    pi_low <- 0.80 + 0.03 * ez
  }
  ifelse(Z == 1, pi_high, pi_low)
}

true_nuisance <- function(X, scenario) {
  e <- prob_z(X, scenario)
  pi <- matrix(0, nrow = nrow(X), ncol = K)
  for (z in 0:(K - 1)) pi[, z + 1] <- true_pi(z, scenario)
  p_r1_x <- rowSums(e * pi)
  list(e = e, pi = pi, p_r1_x = p_r1_x, mu = mu_fun(X))
}


oracle_policy_label <- function(X, Y, scenario) {
  e <- prob_z(X, scenario)
  pi <- true_nuisance(X, scenario)$pi
  mu <- mu_fun(X)
  eps <- Y - mu
  phi_eps <- dnorm(eps)
  score_mat <- matrix(0, nrow = nrow(X), ncol = K)
  for (z in 0:(K - 1)) {
    if (scenario == 1) {
      p_r_y <- ifelse(z == 1, 0.995 - 0.005 * exp(-eps^2), 0.4 * exp(-eps^2))
    } else if (scenario == 2) {
      p_r_y <- ifelse(z == 1, 0.9 - 0.03 * exp(-eps^2), 0.4 * exp(-eps^2))
    } else {
      p_r_y <- ifelse(z == 1, 0.90 - 0.03 * exp(-eps^2), 0.80 + 0.03 * exp(-eps^2))
    }
    score_mat[, z + 1] <- p_r_y * phi_eps
  }
  max.col(score_mat, ties.method = "first") - 1
}

fit_policy <- function(X, Y, scenario) {
  policy_label <- oracle_policy_label(X, Y, scenario)
  policy_train <- data.frame(X1 = X[, 1], X2 = X[, 2], X3 = X[, 3], Y = Y)
  if (length(unique(policy_label)) == 1) return(list(type = "constant", label = unique(policy_label)))
  policy_rf <- randomForest(x = policy_train, y = factor(policy_label), ntree = 300, nodesize = 10)
  list(type = "rf", fit = policy_rf)
}

predict_policy <- function(policy_fit, X, Y) {
  if (policy_fit$type == "constant") return(rep(policy_fit$label, nrow(X)))
  policy_new <- data.frame(X1 = X[, 1], X2 = X[, 2], X3 = X[, 3], Y = Y)
  as.integer(as.character(predict(policy_fit$fit, newdata = policy_new)))
}

fit_scenario <- function(scenario) {
  dat_train <- generate_data(n_train, scenario)
  idx_obs <- dat_train$R == 1
  fit_mu <- randomForest(x = as.data.frame(dat_train$X[idx_obs, , drop = FALSE]), y = dat_train$Y[idx_obs], ntree = 500, nodesize = 5)
  policy_fit <- fit_policy(dat_train$X, dat_train$Y, scenario)
  list(fit_mu = fit_mu, policy_fit = policy_fit)
}

predict_mu <- function(fit_mu, X) {
  as.numeric(predict(fit_mu, newdata = as.data.frame(X)))
}

generate_Y_given_XZ <- function(X, z, scenario, n_mc) {
  X_rep <- X[rep(seq_len(nrow(X)), each = n_mc), , drop = FALSE]
  eps <- rnorm(nrow(X_rep))
  if (scenario == 1) {
    p_r <- ifelse(z == 1, 0.995 - 0.005 * exp(-eps^2), 0.4 * exp(-eps^2))
  } else if (scenario == 2) {
    p_r <- ifelse(z == 1, 0.9 - 0.03 * exp(-eps^2), 0.4 * exp(-eps^2))
  } else {
    p_r <- ifelse(z == 1, 0.90 - 0.03 * exp(-eps^2), 0.80 + 0.03 * exp(-eps^2))
  }
  Y <- mu_fun(X_rep) + eps
  list(X = X_rep, Y = Y, p_r = p_r)
}

compute_m_policy_oracle <- function(X, t_grid, fit_mu, policy_fit, scenario, n_mc = 50) {
  n <- nrow(X)
  nui <- true_nuisance(X, scenario)
  mu_hat <- predict_mu(fit_mu, X)
  M <- array(0, dim = c(n, K, length(t_grid)))
  for (z in 0:(K - 1)) {
    sim <- generate_Y_given_XZ(X, z, scenario, n_mc)
    A <- predict_policy(policy_fit, sim$X, sim$Y)
    S <- abs(sim$Y - rep(mu_hat, each = n_mc))
    for (i in seq_len(n)) {
      ii <- ((i - 1) * n_mc + 1):(i * n_mc)
      M[i, z + 1, ] <- nui$pi[i, z + 1] * colMeans(outer(S[ii], t_grid, "<=") * (A[ii] == z))
      #M[i, z + 1, ] <- colMeans(outer(S[ii], t_grid, "<=") *  (A[ii] == z) * sim$p_r[ii])
    }
  }
  M
}

compute_eif_oracle_policy <- function(dat, t_grid, fit_mu, policy_fit, scenario, n_mc = 50) {
  X <- dat$X
  n <- nrow(X)
  nui <- true_nuisance(X, scenario)
  M <- compute_m_policy_oracle(X, t_grid, fit_mu, policy_fit, scenario, n_mc)
  mu_hat <- predict_mu(fit_mu, X)
  S <- abs(dat$Y - mu_hat)
  A <- predict_policy(policy_fit, X, dat$Y)
  EIF <- matrix(0, nrow = n, ncol = length(t_grid))
  for (i in seq_len(n)) {
    z <- dat$Z[i]
    H <- dat$R[i] * (A[i] == z) * (S[i] <= t_grid)
    EIF[i, ] <- 1 - apply(M[i, , , drop = FALSE], 3, sum) - (H - M[i, z + 1, ]) / nui$e[i, z + 1]
  }
  colMeans(EIF)
}

compute_wcp_threshold <- function(S_cal, X_cal, X_test, scenario, alpha) {
  nui_cal <- true_nuisance(X_cal, scenario)
  nui_test <- true_nuisance(X_test, scenario)
  w_cal <- 1 / pmax(nui_cal$p_r1_x, 1e-8)
  w_test <- 1 / pmax(nui_test$p_r1_x, 1e-8)
  ord <- order(S_cal)
  S_ord <- S_cal[ord]
  w_ord <- w_cal[ord]
  cw <- cumsum(w_ord)
  W <- sum(w_ord)
  threshold <- numeric(nrow(X_test))
  for (i in seq_len(nrow(X_test))) {
    target <- (1 - alpha) * (W + w_test[i])
    j <- which(cw >= target)[1]
    threshold[i] <- if (is.na(j)) max(S_ord) else S_ord[j]
  }
  threshold
}

fits <- vector("list", 3)
for (scenario in 1:3) {
  cat("Fitting scenario", scenario, "\n")
  fits[[scenario]] <- fit_scenario(scenario)
}

run_one <- function(b, scenario, fit_mu, policy_fit) {
  dat_cal <- generate_data(n_cal, scenario)
  dat_test <- generate_data(n_test, scenario)
  X_test <- dat_test$X
  Y_test <- dat_test$Y
  mu_test_hat <- predict_mu(fit_mu, X_test)
  S_test <- abs(Y_test - mu_test_hat)
  eif <- compute_eif_oracle_policy(dat_cal, t_grid, fit_mu, policy_fit, scenario, n_mc)
  out <- numeric(4 * length(alpha_set))
  idx_obs <- dat_cal$R == 1
  mu_cal_hat <- predict_mu(fit_mu, dat_cal$X)
  S_cal <- abs(dat_cal$Y - mu_cal_hat)
  S_cal_obs <- S_cal[idx_obs]
  X_cal_obs <- dat_cal$X[idx_obs, , drop = FALSE]
  for (j in seq_along(alpha_set)) {
    alpha <- alpha_set[j]
    idx <- which(eif >= alpha)
    t_policy <- if(length(idx)==0) {
      -Inf
    } else if(max(idx)==length(t_grid)) {
      Inf
    } else {
      t_grid[max(idx)]
    }
    
    out[4 * j - 3] <- mean(S_test <= t_policy)
    out[4 * j - 2] <- 2 * t_policy
    
    t_wcp <- compute_wcp_threshold(S_cal_obs, X_cal_obs, X_test, scenario, alpha)
    out[4 * j - 1] <- mean(S_test <= t_wcp)
    out[4 * j] <- mean(2 * t_wcp)
  }
  out
}

cl <- makeCluster(n_cores)
clusterEvalQ(cl, library(randomForest))
clusterExport(cl, varlist = c("p", "K", "n_cal", "n_test", "alpha_set", "t_grid", "n_mc", "Sigma", "z_base", "z_x1", "z_x2", "z_x1_3", "z_x2_3", "mu_fun", "prob_z_fun", "prob_z_fun_3", "generate_data_1", "generate_data_2", "generate_data_3", "generate_data", "prob_z", "true_pi", "true_nuisance", "oracle_policy_label", "predict_policy", "generate_Y_given_XZ", "predict_mu", "compute_m_policy_oracle", "compute_eif_oracle_policy", "compute_wcp_threshold", "run_one"), envir = .GlobalEnv)

run_scenario <- function(scenario, fit_mu, policy_fit) {
  res <- pblapply(seq_len(tn), run_one, scenario = scenario, fit_mu = fit_mu, policy_fit = policy_fit, cl = cl)
  do.call(rbind, res)
}

results_all <- vector("list", 3)
for (scenario in 1:3) {
  cat("\nRunning scenario", scenario, "\n")
  results_all[[scenario]] <- run_scenario(scenario, fits[[scenario]]$fit_mu, fits[[scenario]]$policy_fit)
}

stopCluster(cl)


#output_dir <- ""
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

#saveRDS(list(results = results_all, Gamma_hat = gamma_hat), file.path(output_dir, "policy_wcp_three_DGPs.rds"))

for (scenario in 1:3) {
  results <- results_all[[scenario]]
  policy_cov <- results[, seq(1, ncol(results), by = 4), drop = FALSE]
  policy_width <- results[, seq(2, ncol(results), by = 4), drop = FALSE]
  wcp_cov <- results[, seq(3, ncol(results), by = 4), drop = FALSE]
  wcp_width <- results[, seq(4, ncol(results), by = 4), drop = FALSE]
  summary <- rbind(colMeans(policy_cov), apply(policy_cov, 2, sd) / sqrt(tn), colMeans(policy_width), apply(policy_width, 2, sd) / sqrt(tn), colMeans(wcp_cov), apply(wcp_cov, 2, sd) / sqrt(tn), colMeans(wcp_width), apply(wcp_width, 2, sd) / sqrt(tn))
  write.table(summary, file.path(output_dir, paste0("policy_wcp_DGP", scenario, ".txt")), row.names = FALSE)
}


col1 <- "#0055A4"
col2 <- "#D6452D"

par(mfrow=c(2,3), cex.lab=2.5, cex.axis=2, cex.main=2.5, font.main=1, mar=c(5,6,1,0), xpd=NA, oma=c(0,0,2.5,4), mgp=c(3,1,0))

res1 <- results_all[[1]]
policy_cov1 <- colMeans(res1[, seq(1, ncol(res1), by=4), drop=FALSE])
wcp_cov1 <- colMeans(res1[, seq(3, ncol(res1), by=4), drop=FALSE])
policy_se1 <- apply(res1[, seq(1, ncol(res1), by=4), drop=FALSE], 2, sd) / sqrt(tn)
wcp_se1 <- apply(res1[, seq(3, ncol(res1), by=4), drop=FALSE], 2, sd) / sqrt(tn)

matplot(1-alpha_set, cbind(policy_cov1, wcp_cov1), pch=16, ylim=c(0.55,1), xlab=expression(1-alpha), ylab="Coverage rate", cex=1.75, col=c(col1,col2))
title("Setting 1", line=0.75)
arrows(1-alpha_set, policy_cov1-policy_se1, 1-alpha_set, policy_cov1+policy_se1, length=0, angle=90, code=3, lwd=2, col=col1)
arrows(1-alpha_set, wcp_cov1-wcp_se1, 1-alpha_set, wcp_cov1+wcp_se1, length=0, angle=90, code=3, lwd=2, col=col2)
segments(0.7,0.7,0.95,0.95,lty=2,lwd=2)
legend("bottomright", legend=c("IV","WCP"), col=c(col1,col2), pch=16, cex=2, bty="n")

res2 <- results_all[[2]]
policy_cov2 <- colMeans(res2[, seq(1, ncol(res2), by=4), drop=FALSE])
wcp_cov2 <- colMeans(res2[, seq(3, ncol(res2), by=4), drop=FALSE])
policy_se2 <- apply(res2[, seq(1, ncol(res2), by=4), drop=FALSE], 2, sd) / sqrt(tn)
wcp_se2 <- apply(res2[, seq(3, ncol(res2), by=4), drop=FALSE], 2, sd) / sqrt(tn)

matplot(1-alpha_set, cbind(policy_cov2, wcp_cov2), pch=16, ylim=c(0.55,1), xlab=expression(1-alpha), ylab="", cex=1.75, col=c(col1,col2))
title("Setting 2", line=0.75)
arrows(1-alpha_set, policy_cov2-policy_se2, 1-alpha_set, policy_cov2+policy_se2, length=0, angle=90, code=3, lwd=2, col=col1)
arrows(1-alpha_set, wcp_cov2-wcp_se2, 1-alpha_set, wcp_cov2+wcp_se2, length=0, angle=90, code=3, lwd=2, col=col2)
segments(0.7,0.7,0.95,0.95,lty=2,lwd=2)

res3 <- results_all[[3]]
policy_cov3 <- colMeans(res3[, seq(1, ncol(res3), by=4), drop=FALSE])
wcp_cov3 <- colMeans(res3[, seq(3, ncol(res3), by=4), drop=FALSE])
policy_se3 <- apply(res3[, seq(1, ncol(res3), by=4), drop=FALSE], 2, sd) / sqrt(tn)
wcp_se3 <- apply(res3[, seq(3, ncol(res3), by=4), drop=FALSE], 2, sd) / sqrt(tn)

matplot(1-alpha_set, cbind(policy_cov3, wcp_cov3), pch=16, ylim=c(0.55,1), xlab=expression(1-alpha), ylab="", cex=1.75, col=c(col1,col2))
title("Setting 3", line=0.75)
arrows(1-alpha_set, policy_cov3-policy_se3, 1-alpha_set, policy_cov3+policy_se3, length=0, angle=90, code=3, lwd=2, col=col1)
arrows(1-alpha_set, wcp_cov3-wcp_se3, 1-alpha_set, wcp_cov3+wcp_se3, length=0, angle=90, code=3, lwd=2, col=col2)
segments(0.7,0.7,0.95,0.95,lty=2,lwd=2)
res1 <- results_all[[1]]
policy_width1 <- apply(res1[, seq(2, ncol(res1), by=4), drop=FALSE], 2, median, na.rm=TRUE)
wcp_width1 <- apply(res1[, seq(4, ncol(res1), by=4), drop=FALSE], 2, median, na.rm=TRUE)
policy_width1[!is.finite(policy_width1)] <- NA
wcp_width1[!is.finite(wcp_width1)] <- NA

matplot(1-alpha_set, cbind(policy_width1, wcp_width1), pch=16, type="p", ylim=c(1,6), xlab=expression(1-alpha), ylab="Width", cex=1.5, col=c(col1,col2))

res2 <- results_all[[2]]
policy_width2 <- apply(res2[, seq(2, ncol(res2), by=4), drop=FALSE], 2, median, na.rm=TRUE)
wcp_width2 <- apply(res2[, seq(4, ncol(res2), by=4), drop=FALSE], 2, median, na.rm=TRUE)
policy_width2[!is.finite(policy_width2)] <- NA
wcp_width2[!is.finite(wcp_width2)] <- NA

matplot(1-alpha_set, cbind(policy_width2, wcp_width2), pch=16, type="p", ylim=c(1,6), xlab=expression(1-alpha), ylab="", cex=1.5, col=c(col1,col2))

res3 <- results_all[[3]]
policy_width3 <- apply(res3[, seq(2, ncol(res3), by=4), drop=FALSE], 2, median, na.rm=TRUE)
wcp_width3 <- apply(res3[, seq(4, ncol(res3), by=4), drop=FALSE], 2, median, na.rm=TRUE)
policy_width3[!is.finite(policy_width3)] <- NA
wcp_width3[!is.finite(wcp_width3)] <- NA

matplot(1-alpha_set, cbind(policy_width3, wcp_width3), pch=16, type="p", ylim=c(1,6), xlab=expression(1-alpha), ylab="", cex=1.5, col=c(col1,col2))
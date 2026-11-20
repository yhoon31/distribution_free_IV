
set.seed(20261021)
Gamma <- 0.95
rho <- 0.3
epsilon <- 0.01
kappa_min <- 0.4
n_train <- 10000
n_fit <- floor(n_train/2)
n_mass <- n_train-n_fit
n_cal <- 500
n_test <- 500
tn <- 200
alpha_set <- seq(0.05,0.30,by=0.05)
t_grid <- seq(0,8,by=0.02)

true_e1 <- function(D) 0.5+0.1*tanh(D$X1)
make_data <- function(n)
{
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  mu <- X1+0.5*X2
  Y <- mu+rnorm(n)
  e1 <- 0.5+0.1*tanh(X1)
  Z <- rbinom(n,1,e1)
  ez <- ifelse(Z==1,e1,1-e1)
  w <- ez/pmax(e1,1-e1)
  pi <- Gamma*w
  pR <- pi+rho*w*(Gamma-pi)*ifelse(Y>mu,1,-1)
  stopifnot(all(pR>=0 & pR<=1))
  R <- rbinom(n,1,pR)
  data.frame(X1,X2,Y,Z,R)
}
D_fit <- make_data(n_fit)
D_mass <- make_data(n_mass)
fit_mu <- lm(Y~X1+X2,data=D_fit[D_fit$R==1,])
score <- function(D) abs(D$Y-as.numeric(predict(fit_mu,newdata=D)))
fit_e <- glm(Z~X1+X2,data=D_fit,family=binomial())
policy <- function(D) as.integer(predict(fit_e,newdata=D,type='response')>=0.5)
policy_kappa <- function(D) ifelse(policy(D)==1,true_e1(D),1-true_e1(D))


fit_pi <- lapply(0:1,function(z) glm(R~X1+X2,data=D_fit[D_fit$Z==z,],family=binomial()))
fit_f <- lapply(0:1,function(z) lm(Y~X1+X2,data=D_fit[D_fit$Z==z & D_fit$R==1,]))
sd_f <- sapply(fit_f,function(f) summary(f)$sigma)
pred_nui <- function(D)
{
  n <- nrow(D)
  pi <- sapply(1:2,function(j) as.numeric(predict(fit_pi[[j]],newdata=D,type='response')))
  fm <- sapply(1:2,function(j) as.numeric(predict(fit_f[[j]],newdata=D)))
  list(pi=pmin(pmax(pi,1e-5),1),fm=matrix(fm,n,2),
       e=cbind(1-true_e1(D),true_e1(D)))
}

W_mass <- D_mass$R*(D_mass$Z==policy(D_mass))/policy_kappa(D_mass)
theta_hat <- mean(W_mass)
penalty <- sqrt(log(1/epsilon)/(2*n_mass))/kappa_min
theta_L <- max(0,theta_hat-penalty)
cat('theta_hat =',theta_hat,'; penalty =',penalty,'; theta_L =',theta_L,'\n')

wcp_threshold <- function(scores,weights,new_weights,level)
{
  if(!length(scores)) return(rep(Inf,length(new_weights)))
  o <- order(scores); cs <- cumsum(weights[o]); total <- sum(weights)
  target <- level*(total+new_weights)
  ix <- findInterval(target,cs)+1L
  ix <- vapply(target,function(t) { j <- which(cs>=t)[1L]; if(is.na(j)) length(cs)+1L else j },integer(1))
  out <- rep(Inf,length(ix)); good <- ix<=length(cs)
  out[good] <- scores[o][ix[good]]; out
}

eif_curve <- function(D)
{
  nui <- pred_nui(D)
  n <- nrow(D); a <- policy(D)
  ix <- cbind(seq_len(n),a+1L)
  pi_a <- nui$pi[ix]
  fm_a <- nui$fm[ix]
  muhat <- as.numeric(predict(fit_mu,newdata=D))
  s <- abs(D$Y-muhat)
  ez <- nui$e[cbind(seq_len(n),D$Z+1L)]
  matched <- (D$Z==a)
  out <- numeric(length(t_grid))
  for(j in seq_along(t_grid)) {
    t <- t_grid[j]
    prob <- pnorm(muhat+t,mean=fm_a,sd=sd_f[a+1L])-
            pnorm(muhat-t,mean=fm_a,sd=sd_f[a+1L])
    M <- pi_a*prob
    out[j] <- mean(1-M-(D$R*matched*(s<=t)-matched*M)/ez)
  }
  out
}

one_trial <- function(i)
{
  cal <- make_data(n_cal)
  test <- make_data(n_test)
  s_test <- score(test)
  a_cal <- policy(cal)
  matched <- cal$R==1 & cal$Z==a_cal
  s_cal <- score(cal[matched,,drop=FALSE]);
  w_cal <- 1/policy_kappa(cal[matched,,drop=FALSE])
  w_test <- 1/policy_kappa(test)
  delta <- eif_curve(cal)
  out <- matrix(NA_real_,nrow=length(alpha_set),ncol=5)
  for(j in seq_along(alpha_set)) {
    alpha <- alpha_set[j]
    level <- (1-alpha)/((1-epsilon)*theta_L)
    if(!is.finite(level) || level>=1 || !length(s_cal)) {
      tw <- rep(Inf,n_test)
    } else tw <- wcp_threshold(s_cal,w_cal,w_test,level)
    ids <- which(delta >= alpha)
    te <- if(length(ids)==0) {
      -Inf
    } else if(max(ids)==length(t_grid)) {
      Inf
    } else {
      t_grid[max(ids)+1L]
    }
    out[j,] <- c(mean(s_test<=tw),median(2*tw),mean(s_test<=te),2*te,length(s_cal))
  }
  out
}

library(parallel)
library(pbapply)
n_cores <- max(1L, parallel::detectCores() - 1L)
cl <- parallel::makeCluster(n_cores)
parallel::clusterEvalQ(cl, {
  library(randomForest)
  library(nnet)
  NULL
})
parallel::clusterExport(cl,varlist = setdiff(ls(envir = .GlobalEnv), c("cl", "results")),envir = .GlobalEnv)
results <- tryCatch(pbapply::pblapply(seq_len(tn), one_trial, cl = cl),finally = parallel::stopCluster(cl))
res <- do.call(rbind,lapply(seq_along(alpha_set),function(j){
  v <- do.call(rbind,lapply(results,function(x)x[j,]))
  data.frame(alpha=alpha_set[j],target=1-alpha_set[j],
             WCP_coverage=mean(v[,1]),WCP_se=sd(v[,1])/sqrt(tn),
             WCP_width_median=median(v[,2]),
             EIF_coverage=mean(v[,3]),EIF_se=sd(v[,3])/sqrt(tn),
             EIF_width_median=median(v[,4]),matched_mean=mean(v[,5]))
}))
print(res,row.names=FALSE)

#write.table(res, paste0("known_pzx_K_2_ntrain_", n_train, ".txt"))

matplot(1-alpha_set,res[,c('WCP_coverage','EIF_coverage')],type='b',pch=c(16,17),
        col=c('#D95F5F','#4F81BD'),ylim=c(0.6,1),xlab=expression(1-alpha),ylab='Coverage')
abline(0,1,lty=2)
legend('bottomright',c('WCP','EIF (true P(Z|X))'),pch=c(16,17), col=c('#D95F5F','#4F81BD'),bty='n')

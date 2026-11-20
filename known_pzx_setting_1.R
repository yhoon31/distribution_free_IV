
library(randomForest)
library(nnet)
library(parallel)
library(pbapply)

p <- 10
K <- 10
Gamma <- 0.95
mu_coef <- c(1,0.8,0.6,0.4,0.2,-0.2,-0.4,-0.6,-0.8,-1)
base_prob <- c(0.4,rep(0.6/(K-1),K-1))
theta <- matrix(0,nrow=K,ncol=p)
theta[1,1:3] <- c(0.16,-0.11,0.08)
for(z in 2:K) theta[z,1:3] <- c(-0.05,0.04,-0.03)*(z-1)/(K-1)
kappa_min <- 0.05
mix_prob <- 0.9
shift_left <- -1
shift_right <- 9
y_sd <- 0.25
rho <- 0.2
Eh <- 1-2*mix_prob

make_X_df <- function(X){X <- as.matrix(X); storage.mode(X) <- "double"; X_df <- as.data.frame(X,stringsAsFactors=FALSE); colnames(X_df) <- paste0("X",seq_len(ncol(X_df))); X_df}
softmax_mat <- function(eta){eta <- eta-apply(eta,1,max); ee <- exp(eta); ee/rowSums(ee)}
get_e_true <- function(X){
  X <- as.matrix(X)
  eta <- sweep(X %*% t(theta),2,log(base_prob),"+")
  0.05 + 0.5 * softmax_mat(eta)
}

generate_data <- function(n,Gamma){
  X <- matrix(rnorm(n*p),nrow=n,ncol=p)
  e <- get_e_true(X)
  Z <- apply(e,1,function(pr) sample.int(K,1,prob=pr))
  w <- e/apply(e,1,max)
  pi_z <- Gamma*w[cbind(seq_len(n),Z)]
  mu <- as.vector(X%*%mu_coef)
  B <- rbinom(n,1,mix_prob)
  Y <- mu+ifelse(B==1,shift_left,shift_right)+y_sd*rnorm(n)
  h <- ifelse(Y>mu,1,-1)
  pR <- pi_z+rho*(pi_z/Gamma)*(Gamma-pi_z)*(h-Eh)
  if(any(pR<0|pR>1)) stop("pR outside [0,1]")
  R <- rbinom(n,1,pR)
  list(X=X,Z=Z,R=R,Y=Y)
}


set.seed(20261021)
n_train <- 10000
n_fit <- floor(n_train/2)
n_mass <- n_train - n_fit

dat_train <- generate_data(n_fit,Gamma)
X_train <- as.matrix(dat_train$X)
X_train_df <- make_X_df(X_train)
idx_obs <- dat_train$R==1

fit_mu <- randomForest(x=X_train[idx_obs,,drop=FALSE],y=dat_train$Y[idx_obs],ntree=500,nodesize=5)
predict_mu_hat <- function(X) as.numeric(predict(fit_mu,newdata=as.matrix(X)))

fit_e <- multinom(Z~.,data=data.frame(X_train_df,Z=factor(dat_train$Z,levels=seq_len(K))),trace=FALSE,maxit=300,MaxNWts=10000)

fit_pi <- vector("list",K)
for(z in seq_len(K))
{
  idx_z <- which(dat_train$Z==z)
  rz <- dat_train$R[idx_z]
  if(length(idx_z)==0) fit_pi[[z]] <- list(constant=mean(dat_train$R))
  else if(length(unique(rz))<2) fit_pi[[z]] <- list(constant=mean(rz))
  else fit_pi[[z]] <- randomForest(x=X_train[idx_z,,drop=FALSE],y=factor(rz,levels=c(0,1)),ntree=300,nodesize=10)
}

fit_f_mean <- vector("list",K)
fit_f_sq <- vector("list",K)
for(z in seq_len(K))
{
  idx_z <- which(dat_train$Z==z&dat_train$R==1)
  if(length(idx_z)<5) idx_z <- which(dat_train$R==1)
  fit_f_mean[[z]] <- randomForest(x=X_train[idx_z,,drop=FALSE],y=dat_train$Y[idx_z],ntree=300,nodesize=10)
  fit_f_sq[[z]] <- randomForest(x=X_train[idx_z,,drop=FALSE],y=dat_train$Y[idx_z]^2,ntree=300,nodesize=10)
}

predict_nuisance <- function(X){
  X <- as.matrix(X); n <- nrow(X)
  e_raw <- as.matrix(predict(fit_e,newdata=make_X_df(X),type="probs"))
  if(n==1&&nrow(e_raw)!=1) e_raw <- matrix(e_raw,nrow=1)
  e_hat <- matrix(1e-8,nrow=n,ncol=K)
  if(!is.null(colnames(e_raw))){
    cols <- suppressWarnings(as.integer(colnames(e_raw)))
    good <- which(!is.na(cols)&cols>=1&cols<=K)
    if(length(good)>0) e_hat[,cols[good]] <- e_raw[,good,drop=FALSE]
  } else e_hat[,seq_len(min(K,ncol(e_raw)))] <- e_raw[,seq_len(min(K,ncol(e_raw))),drop=FALSE]
  e_hat <- pmax(e_hat,1e-8); e_hat <- e_hat/rowSums(e_hat)
  pi_hat <- matrix(0,nrow=n,ncol=K)
  for(z in seq_len(K)){
    if(!is.null(fit_pi[[z]]$constant)) pi_hat[,z] <- fit_pi[[z]]$constant
    else{
      pr <- predict(fit_pi[[z]],newdata=X,type="prob")
      pi_hat[,z] <- if("1"%in%colnames(pr)) pr[,"1"] else pr[,ncol(pr)]
    }
  }
  pi_hat <- pmin(pmax(pi_hat,1e-6),1)
  mu_f <- matrix(0,nrow=n,ncol=K); var_f <- matrix(0,nrow=n,ncol=K)
  for(z in seq_len(K)){
    mu_f[,z] <- predict(fit_f_mean[[z]],newdata=X)
    EY2 <- predict(fit_f_sq[[z]],newdata=X)
    var_f[,z] <- pmax(EY2-mu_f[,z]^2,0.05^2)
  }
  list(e_hat=e_hat,pi_hat=pi_hat,mu_f=mu_f,sd_f=sqrt(var_f))
}

true_nuisance <- function(X,Gamma){
  X <- as.matrix(X); e <- get_e_true(X); w <- e/apply(e,1,max); pi <- Gamma*w; mu <- as.vector(X%*%mu_coef)
  list(e=e,pi=pi,mu=mu,w=w)
}

true_y_density_vec <- function(y,mu) mix_prob*dnorm(y,mean=mu+shift_left,sd=y_sd)+(1-mix_prob)*dnorm(y,mean=mu+shift_right,sd=y_sd)
true_f_density_vec <- function(y,mu,pi,Gamma){h <- ifelse(y>mu,1,-1); true_y_density_vec(y,mu)*(1+rho*(Gamma-pi)/Gamma*(h-Eh))}
true_q_density_vec <- function(y,mu,pi,Gamma){h <- ifelse(y>mu,1,-1); true_y_density_vec(y,mu)*(pi+rho*(pi/Gamma)*(Gamma-pi)*(h-Eh))}


nuisance_train <- predict_nuisance(X_train)
qhat_train <- matrix(0,nrow=sum(idx_obs),ncol=K)
for(z in seq_len(K)) qhat_train[,z] <- nuisance_train$pi_hat[idx_obs,z]*dnorm(dat_train$Y[idx_obs],mean=nuisance_train$mu_f[idx_obs,z],sd=nuisance_train$sd_f[idx_obs,z])

policy_label <- max.col(qhat_train,ties.method="first")
policy_levels <- sort(unique(policy_label))
if(length(policy_levels)==1) policy_fit <- list(constant=policy_levels[1]) else policy_fit <- randomForest(x=data.frame(make_X_df(X_train[idx_obs,,drop=FALSE]),Y=dat_train$Y[idx_obs]),y=factor(policy_label,levels=policy_levels),ntree=300,nodesize=10)
predict_policy <- function(policy_fit,X,Y){X <- as.matrix(X); if(!is.null(policy_fit$constant)) rep(policy_fit$constant,nrow(X)) else as.integer(as.character(predict(policy_fit,newdata=data.frame(make_X_df(X),Y=Y))))}

epsilon <- 0.01
dat_mass <- generate_data(n_mass,Gamma)
A_mass <- predict_policy(policy_fit,dat_mass$X,dat_mass$Y)
e_mass <- get_e_true(dat_mass$X)
W_mass <- as.numeric(dat_mass$R==1 & dat_mass$Z==A_mass) /
  e_mass[cbind(seq_len(n_mass),dat_mass$Z)]
theta_lower <- max(0, mean(W_mass) - (1/kappa_min)*sqrt(log(1/epsilon)/(2*n_mass)))
cat('theta lower bound:',theta_lower,'; mean W:',mean(W_mass), '\n')
cat('Hoeffding penalty:',(1/kappa_min)*sqrt(log(1/epsilon)/(2*n_mass)), '\n')
if(theta_lower <= 0) message('The mass lower bound is zero: the valid WCP prediction set is R for every alpha.')

weighted_thresholds <- function(scores,weights,w_test,level){
  if(length(scores)==0L) return(rep(Inf,length(w_test)))
  ord <- order(scores)
  scores <- scores[ord]; weights <- weights[ord]
  cs <- cumsum(weights); total <- sum(weights)
  sapply(w_test,function(w){
    target <- level*(total+w)
    if(target > total) return(Inf)
    scores[which(cs>=target)[1L]]
  })
}

wcp_eval <- function(dat_cal,X_test,Y_test,alpha_set,theta_lower,
                     width_n=50L,grid_step=0.1,grid_radius=20){
  n_test <- nrow(X_test)
  A_cal <- predict_policy(policy_fit,dat_cal$X,dat_cal$Y)
  matched <- which(dat_cal$R==1 & dat_cal$Z==A_cal)
  if(length(matched)){
    e_cal <- get_e_true(dat_cal$X[matched,,drop=FALSE])
    weights <- 1/e_cal[cbind(seq_along(matched),dat_cal$Z[matched])]
    scores <- abs(dat_cal$Y[matched]-predict_mu_hat(dat_cal$X[matched,,drop=FALSE]))
  }else{weights <- numeric(0); scores <- numeric(0)}
  mu_test <- predict_mu_hat(X_test)
  e_test <- get_e_true(X_test)
  A_test <- predict_policy(policy_fit,X_test,Y_test)
  score_test <- abs(Y_test-mu_test)
  coverage <- numeric(length(alpha_set))
  widths <- matrix(Inf,nrow=min(width_n,n_test),ncol=length(alpha_set))
  for(j in seq_along(alpha_set)){
    alpha <- alpha_set[j]
    level <- (1-alpha)/((1-epsilon)*theta_lower)
    if(!is.finite(level) || level>=1 || level<=0 || length(matched)==0L){
      coverage[j] <- 1
      next
    }
   
    thresholds <- matrix(Inf,nrow=n_test,ncol=K)
    for(z in seq_len(K)) thresholds[,z] <- weighted_thresholds(
      scores,weights,1/e_test[,z],level)
    coverage[j] <- mean(score_test <= thresholds[cbind(seq_len(n_test),A_test)])

    
    for(i in seq_len(nrow(widths))){
      y_grid <- seq(mu_test[i]-grid_radius,mu_test[i]+grid_radius,by=grid_step)
      x_grid <- matrix(rep(X_test[i,],each=length(y_grid)),ncol=p)
      arms <- predict_policy(policy_fit,x_grid,y_grid)
      included <- abs(y_grid-mu_test[i]) <= thresholds[i,arms]
      if(included[1] || included[length(included)]){
        widths[i,j] <- Inf
      }else{
        widths[i,j] <- sum((as.numeric(included[-1])+as.numeric(included[-length(included)]))/2)*grid_step
      }
    }
  }
  list(coverage=coverage,width=widths,n_matched=length(matched), fallback=as.integer(!is.finite((1-alpha_set)/((1-epsilon)*theta_lower)) | (1-alpha_set)/((1-epsilon)*theta_lower)>=1 | length(matched)==0L))
}

generate_estimated_mc <- function(X,muhat,nuisance,n_mc){
  n <- nrow(X); id_i <- rep(seq_len(n),each=K*n_mc); id_z <- rep(rep(seq_len(K),each=n_mc),times=n)
  mu_vec <- nuisance$mu_f[cbind(id_i,id_z)]; sd_vec <- nuisance$sd_f[cbind(id_i,id_z)]; Y_mc <- mu_vec+sd_vec*rnorm(length(id_i))
  list(id_i=id_i,id_z=id_z,Y=Y_mc,S=abs(Y_mc-muhat[id_i]))
}

compute_M_fast <- function(A_mc,S_mc,id_i,id_z,pi_mat,n,t_grid,n_mc){
  M <- array(0,dim=c(n,K,length(t_grid)))
  for(z in seq_len(K)){
    idx_z <- which(id_z==z); A_z <- A_mc[idx_z]; S_z <- S_mc[idx_z]
    for(i in seq_len(n)){
      idx <- ((i-1)*n_mc+1):(i*n_mc); scores <- sort(S_z[idx][A_z[idx]==z])
      if(length(scores)>0) M[i,z,] <- pi_mat[i,z]*findInterval(t_grid,scores)/n_mc
    }
  }
  M
}


compute_eif_curve <- function(dat,t_grid,M,A_obs,e_mat,muhat){
  n <- nrow(dat$X); z <- dat$Z; e_obs <- e_mat[cbind(seq_len(n),z)]; S_obs <- abs(dat$Y-muhat); M_sum <- apply(M,c(1,3),sum); EIF <- 1-M_sum
  for(i in seq_len(n)){H_i <- dat$R[i]*(S_obs[i]<=t_grid)*(A_obs[i]==z[i]); EIF[i,] <- EIF[i,]-(H_i-M[i,z[i],])/e_obs[i]}
  colMeans(EIF)
}


tn <- 200
n_cal <- 500
n_test <- 500
alpha_set <- seq(0.05,0.30,by=0.05)
t_grid <- seq(0,12,by=0.1)
n_mc <- 20               
n_width <- 20         

one_trial <- function(trial){
  dat_cal <- generate_data(n_cal,Gamma)
  X_cal <- dat_cal$X
  X_test <- matrix(rnorm(n_test*p),nrow=n_test,ncol=p)
  mu_test_true <- as.vector(X_test%*%mu_coef)
  B_test <- rbinom(n_test,1,mix_prob)
  Y_test <- mu_test_true+ifelse(B_test==1,shift_left,shift_right)+y_sd*rnorm(n_test)
  S_test <- abs(Y_test-predict_mu_hat(X_test))


  wc <- wcp_eval(dat_cal,X_test,Y_test,alpha_set,theta_lower,width_n=n_width)

  nuisance_cal <- predict_nuisance(X_cal)
  e_true_cal <- get_e_true(X_cal)
  muhat_cal <- predict_mu_hat(X_cal)
  mc <- generate_estimated_mc(X_cal,muhat_cal,nuisance_cal,n_mc)
  A_mc <- predict_policy(policy_fit,X_cal[mc$id_i,,drop=FALSE],mc$Y)
  M <- compute_M_fast(A_mc,mc$S,mc$id_i,mc$id_z,
                      nuisance_cal$pi_hat,n_cal,t_grid,n_mc)
  A_obs <- predict_policy(policy_fit,X_cal,dat_cal$Y)
  delta <- compute_eif_curve(dat_cal,t_grid,M,A_obs,e_true_cal,muhat_cal)

  out <- matrix(NA_real_,nrow=length(alpha_set),ncol=6)
  for(j in seq_along(alpha_set))
  {
    alpha <- alpha_set[j]
    ids <- which(delta >= alpha)
    t_eif <- if(length(ids)==0) {
      -Inf
    } else if(max(ids)==length(t_grid)) {
      Inf
    } else {
      t_grid[max(ids)+1]
    }
    
    out[j,] <- c(wc$coverage[j],median(wc$width[,j]),
                mean(S_test<=t_eif),2*t_eif,wc$n_matched,wc$fallback[j])
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
parallel::clusterExport(cl, varlist = setdiff(ls(envir = .GlobalEnv), c("cl", "results")),envir = .GlobalEnv)
results <- tryCatch(pbapply::pblapply(seq_len(tn), one_trial, cl = cl),finally = parallel::stopCluster(cl))
res <- do.call(rbind,lapply(seq_along(alpha_set),function(j){
  v <- do.call(rbind,lapply(results,function(x)x[j,]))
  data.frame(alpha=alpha_set[j],target=1-alpha_set[j],
    WCP_coverage=mean(v[,1]),WCP_se=sd(v[,1])/sqrt(tn),
    WCP_width_median=median(v[,2]),
    EIF_coverage=mean(v[,3]),EIF_se=sd(v[,3])/sqrt(tn),
    EIF_width_median=median(v[,4]),
    matched_mean=mean(v[,5]),WCP_fallback_fraction=mean(v[,6]))
}))
print(res,row.names=FALSE)

#write.table(res, paste0("known_pzx_K_10_ntrain_", n_train, ".txt"))

matplot(1-alpha_set,res[,c('WCP_coverage','EIF_coverage')],
        type='b',pch=c(16,17),col=c('#D95F5F','#4F81BD'),
        ylim=c(0.6,1),xlab=expression(1-alpha),ylab='Coverage rate')
abline(0,1,lty=2)
legend('bottomright',c('WCP','EIF (true P(Z|X))'),
       pch=c(16,17),col=c('#D95F5F','#4F81BD'),bty='n')

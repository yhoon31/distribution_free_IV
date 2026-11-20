library(parallel)
library(pbapply)

p <- 10
K <- 10
theta <- matrix(seq(-1.5,1.5,length.out=K*p),nrow=K,ncol=p)
mu_coef <- c(1,0.8,0.6,0.4,0.2,-0.2,-0.4,-0.6,-0.8,-1)
Gamma <- 0.8
lambda_iv <- 0.2
z_effect <- seq(-1,1,length.out=K)

generate_data <- function(n,Gamma,lambda_iv){
  X <- matrix(rnorm(n*p),nrow=n,ncol=p)
  eta <- 0.5*X%*%t(theta)
  e <- exp(eta-apply(eta,1,max))
  e <- e/rowSums(e)
  Z <- apply(e,1,function(prob) sample.int(K,1,prob=prob))
  w <- e/apply(e,1,max)
  pi_z <- Gamma*w[cbind(seq_len(n),Z)]
  mu <- as.vector(X%*%mu_coef)
  B <- sample(c(-1,1),n,replace=TRUE)
  center <- mu+lambda_iv*z_effect[Z]
  Y <- center+2.5*B+0.25*rnorm(n)
  h <- ifelse(Y>center,1,-1)
  pR <- pi_z+0.8*(pi_z/Gamma)*(Gamma-pi_z)*h
  R <- rbinom(n,1,pR)
  list(X=X,Z=Z,R=R,Y=Y)
}

true_nuisance <- function(X,Gamma){
  X <- as.matrix(X)
  eta <- 0.5*X%*%t(theta)
  e <- exp(eta-apply(eta,1,max))
  e <- e/rowSums(e)
  w <- e/apply(e,1,max)
  pi <- Gamma*w
  mu <- as.vector(X%*%mu_coef)
  list(e=e,pi=pi,mu=mu)
}

generate_true_mc <- function(X,Gamma,lambda_iv,n_mc){
  n <- nrow(X)
  nui <- true_nuisance(X,Gamma)
  id_i <- rep(seq_len(n),each=K*n_mc)
  id_z <- rep(rep(seq_len(K),each=n_mc),times=n)
  pi_vec <- nui$pi[cbind(id_i,id_z)]
  center <- nui$mu[id_i]+lambda_iv*z_effect[id_z]
  a <- 0.8*(Gamma-pi_vec)/Gamma
  Y <- numeric(length(id_i))
  accepted <- rep(FALSE,length(id_i))
  while(any(!accepted)){
    idx <- which(!accepted)
    B <- sample(c(-1,1),length(idx),replace=TRUE)
    y_prop <- center[idx]+2.5*B+0.25*rnorm(length(idx))
    h <- ifelse(y_prop>center[idx],1,-1)
    acc <- runif(length(idx))<=(1+a[idx]*h)/(1+a[idx])
    Y[idx[acc]] <- y_prop[acc]
    accepted[idx[acc]] <- TRUE
  }
  list(id_i=id_i,id_z=id_z,Y=Y,S=abs(Y-nui$mu[id_i]))
}

get_true_policy <- function(X,Y,Gamma,lambda_iv){
  X <- as.matrix(X)
  nui <- true_nuisance(X,Gamma)
  n <- nrow(X)
  Q <- matrix(0,nrow=n,ncol=K)
  for(z in seq_len(K)){
    pi_z <- nui$pi[,z]
    center <- nui$mu+lambda_iv*z_effect[z]
    h <- ifelse(Y>center,1,-1)
    f_y <- 0.5*dnorm(Y,center-2.5,0.25)+0.5*dnorm(Y,center+2.5,0.25)
    Q[,z] <- f_y*(pi_z+0.8*(pi_z/Gamma)*(Gamma-pi_z)*h)
  }
  max.col(Q,ties.method="first")
}

compute_M_fast <- function(A_mc,S_mc,id_i,id_z,pi_mat,n,t_grid,n_mc){
  M <- array(0,dim=c(n,K,length(t_grid)))
  for(z in seq_len(K)){
    idx_z <- which(id_z==z)
    A_z <- A_mc[idx_z]
    S_z <- S_mc[idx_z]
    for(i in seq_len(n)){
      idx <- ((i-1)*n_mc+1):(i*n_mc)
      scores <- sort(S_z[idx][A_z[idx]==z])
      if(length(scores)>0) M[i,z,] <- pi_mat[i,z]*findInterval(t_grid,scores)/n_mc
    }
  }
  M
}

compute_oracle_eif <- function(dat,t_grid,Gamma,lambda_iv,n_mc){
  X <- as.matrix(dat$X)
  n <- nrow(X)
  nui <- true_nuisance(X,Gamma)
  mu <- nui$mu
  
  mc <- generate_true_mc(X,Gamma,lambda_iv,n_mc)
  X_mc <- X[mc$id_i,,drop=FALSE]
  A_mc <- get_true_policy(X_mc,mc$Y,Gamma,lambda_iv)
  M <- compute_M_fast(A_mc,mc$S,mc$id_i,mc$id_z,nui$pi,n,t_grid,n_mc)
  
  A_obs <- get_true_policy(X,dat$Y,Gamma,lambda_iv)
  S_obs <- abs(dat$Y-mu)
  e_obs <- nui$e[cbind(seq_len(n),dat$Z)]
  M_sum <- apply(M,c(1,3),sum)
  EIF <- 1-M_sum
  
  for(i in seq_len(n)){
    z <- dat$Z[i]
    H_i <- dat$R[i]*(S_obs[i]<=t_grid)*(A_obs[i]==z)
    EIF[i,] <- EIF[i,]-(H_i-M[i,z,])/e_obs[i]
  }
  colMeans(EIF)
}

get_t_eif <- function(eif_curve,alpha,t_grid){
  valid <- which(eif_curve<=alpha)
  if(length(valid)==0) return(Inf)
  t_grid[min(valid)]
}

tn <- 200
n_cal <- 500
n_test <- 500
alpha_set <- seq(0.05,0.30,by=0.05)
t_grid <- seq(0,8,by=0.02)
n_mc <- 100
n_cores <- max(1,detectCores()-1)

cl <- makeCluster(n_cores)
clusterExport(cl,c("p","K","theta","mu_coef","Gamma","lambda_iv","z_effect","generate_data","true_nuisance","generate_true_mc","get_true_policy","compute_M_fast","compute_oracle_eif","get_t_eif","n_cal","n_test","alpha_set","t_grid","n_mc"),envir=.GlobalEnv)

results_list <- pblapply(seq_len(tn),function(trial){
  dat_cal <- generate_data(n_cal,Gamma,lambda_iv)
  eif_oracle <- compute_oracle_eif(dat_cal,t_grid,Gamma,lambda_iv,n_mc)
  
  dat_test <- generate_data(n_test,Gamma,lambda_iv)
  mu_test <- as.vector(dat_test$X%*%mu_coef)
  S_test <- abs(dat_test$Y-mu_test)
  
  result <- numeric(2*length(alpha_set))
  for(j in seq_along(alpha_set)){
    t_star <- get_t_eif(eif_oracle,alpha_set[j],t_grid)
    result[2*j-1] <- mean(S_test<=t_star)
    result[2*j] <- 2*t_star
  }
  result
},cl=cl)

stopCluster(cl)

results <- do.call(rbind,results_list)
colnames(results) <- unlist(lapply(alpha_set,function(alpha) c(paste0("coverage_alpha_",alpha),paste0("width_alpha_",alpha))))

coverage <- results[,grep("^coverage",colnames(results)),drop=FALSE]
width <- results[,grep("^width",colnames(results)),drop=FALSE]

res <- cbind(
  coverage_mean=colMeans(coverage),
  coverage_se=apply(coverage,2,sd)/sqrt(tn),
  width_mean=colMeans(width),
  width_se=apply(width,2,sd)/sqrt(tn)
)
rownames(res) <- paste0("alpha_",alpha_set)
print(res)

#write.table(res,"invalid_Gamma_08_par_02.txt")

coverage_mean <- res[,"coverage_mean"]
coverage_se <- res[,"coverage_se"]
width_mean <- res[,"width_mean"]
width_se <- res[,"width_se"]

par(mfrow=c(1,2),mgp=c(3.5,1,0),cex.lab=2,cex.axis=1.7,mar=c(2.5,5,4,0),oma=c(2.5,1,0,2),xpd=NA,font.main=1,cex.main=2)

plot(1-alpha_set,coverage_mean,pch=16,xlab=expression(1-alpha),ylab="Coverage rate",lwd=2,ylim=c(0.5,1))
arrows(1-alpha_set,coverage_mean-coverage_se,1-alpha_set,pmin(coverage_mean+coverage_se,1),length=0.01,angle=90,code=3,lwd=2)
segments(min(1-alpha_set),min(1-alpha_set),max(1-alpha_set),max(1-alpha_set),lty=2,lwd=2)

plot(1-alpha_set,width_mean,pch=16,xlab=expression(1-alpha),ylab="Width",lwd=2)
arrows(1-alpha_set,width_mean-width_se,1-alpha_set,width_mean+width_se,length=0.01,angle=90,code=3,lwd=2)

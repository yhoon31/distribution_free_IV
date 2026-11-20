
##invalid IV
res01 <- read.table("invalid_Gamma_1_par_0_new.txt",header=T)[,1:4]
res11 <- read.table("invalid_Gamma_1_par_01_new.txt",header=T)[,1:4]
res21 <- read.table("invalid_Gamma_1_par_02_new.txt",header=T)[,1:4]
res02 <- read.table("invalid_Gamma_09_par_0_new.txt",header=T)
res12 <- read.table("invalid_Gamma_09_par_01_new.txt",header=T)
res22 <- read.table("invalid_Gamma_09_par_02_new.txt",header=T)
res03 <- read.table("invalid_Gamma_08_par_0_new.txt",header=T)
res13 <- read.table("invalid_Gamma_08_par_01_new.txt",header=T)
res23 <- read.table("invalid_Gamma_08_par_02_new.txt",header=T)

col1 <- "#4F81BD"
col2 <- "#58B7B0"
col3 <- "#7F8C8D"

par(mfrow = c(1, 3), mgp = c(3.5, 1, 0), cex.lab = 2.5, cex.axis = 2.5, mar = c(2.5, 5, 4, 0), oma = c(2.5, 1, 0, 2), xpd = NA, font.main = 1, cex.main = 2)
matplot(1 - alpha_set, cbind(res01[,1],res11[,1],res21[,1]), type = "p", pch = 16, xlab = expression(1 - alpha), ylab = "Coverage rate", lwd = 2, ylim = c(0.5, 1),col = c(col1,col2,col3),cex=2,main=expression(paste(Gamma, " = 1")))
arrows(1 - alpha_set, res01[,1] - res01[,2], 1 - alpha_set, pmin(res01[,1] + res01[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col1)
arrows(1 - alpha_set, res11[,1] - res11[,2], 1 - alpha_set, pmin(res11[,1] + res11[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col2)
arrows(1 - alpha_set, res21[,1] - res21[,2], 1 - alpha_set, pmin(res21[,1] + res21[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col3)
segments(min(1 - alpha_set), min(1 - alpha_set), max(1 - alpha_set), max(1 - alpha_set), lty = 2, lwd = 2)

legend("bottomright", legend = c("q = 0", "q = 0.1", "q = 0.2"), col = c(col1,col2,col3), pch = 16, bty = "n", cex = 2)

matplot(1 - alpha_set, cbind(res02[,1],res12[,1],res22[,1]), type = "p", pch = 16, xlab = expression(1 - alpha), ylab = "", lwd = 2, ylim = c(0.5, 1),col = c(col1,col2,col3),cex=2,main=expression(paste(Gamma, " = 0.9")))
arrows(1 - alpha_set, res02[,1] - res02[,2], 1 - alpha_set, pmin(res02[,1] + res02[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col1)
arrows(1 - alpha_set, res12[,1] - res12[,2], 1 - alpha_set, pmin(res12[,1] + res12[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col2)
arrows(1 - alpha_set, res22[,1] - res22[,2], 1 - alpha_set, pmin(res22[,1] + res22[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col3)
segments(min(1 - alpha_set), min(1 - alpha_set), max(1 - alpha_set), max(1 - alpha_set), lty = 2, lwd = 2)

matplot(1 - alpha_set, cbind(res03[,1],res13[,1],res23[,1]), type = "p", pch = 16, xlab = expression(1 - alpha), ylab = "", lwd = 2, ylim = c(0.5, 1),col = c(col1,col2,col3),cex=2,main=expression(paste(Gamma, " = 0.8")))
arrows(1 - alpha_set, res03[,1] - res03[,2], 1 - alpha_set, pmin(res03[,1] + res03[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col1)
arrows(1 - alpha_set, res13[,1] - res13[,2], 1 - alpha_set, pmin(res13[,1] + res13[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col2)
arrows(1 - alpha_set, res23[,1] - res23[,2], 1 - alpha_set, pmin(res23[,1] + res23[,2], 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col3)
segments(min(1 - alpha_set), min(1 - alpha_set), max(1 - alpha_set), max(1 - alpha_set), lty = 2, lwd = 2)



##-----------------------------------------------------



res11 <- read.table("policy_Gamma_1_setting_2_tr_2000_new.txt", header = TRUE)
res12 <- read.table("policy_Gamma_09_setting_2_tr_2000_new.txt", header = TRUE)
res13 <- read.table("policy_Gamma_08_setting_2_tr_2000_new.txt", header = TRUE)
res21 <- read.table("policy_Gamma_1_setting_2_tr_500_new.txt", header = TRUE)
res22 <- read.table("policy_Gamma_09_setting_2_tr_500_new.txt", header = TRUE)
res23 <- read.table("policy_Gamma_08_setting_2_tr_500_new.txt", header = TRUE)

alpha_set <- seq(0.05, 0.30, by = 0.05)

col1 <- "#5A5A5A"
col2 <- "#E9A95B"
col3 <- "#5B9BD5"
col4 <- "#D9796F"

pch1 <- 16
pch2 <- 15
pch3 <- 17
pch4 <- 18

plot_policy <- function(res, Gamma, title = FALSE, ylab = "", xlab = "", show_legend = FALSE) {
  x <- 1 - alpha_set
  offset <- 0.005
  x_oracle <- x - offset
  x_eif_oracle <- x + offset
  
  oracle <- res$true_oracle_coverage_mean
  plugin <- res$plugin_coverage_mean
  eif_oracle <- res$eif_oracle_policy_coverage_mean
  policy <- res$eif_estimated_coverage_mean
  
  oracle_se <- res$true_oracle_coverage_se
  plugin_se <- res$plugin_coverage_se
  eif_oracle_se <- res$eif_oracle_policy_coverage_se
  policy_se <- res$eif_estimated_coverage_se
  
  matplot(x, cbind(oracle, plugin, eif_oracle, policy), type = "n", xlab = xlab, ylab = ylab, ylim = c(0.4, 1), main = if (title) bquote(Gamma == .(Gamma)) else "")
  
  points(x_oracle, oracle, pch = pch1, col = col1, cex = 1.5)
  points(x, plugin, pch = pch2, col = col2, cex = 1.5)
  points(x_eif_oracle, eif_oracle, pch = pch3, col = col3, cex = 1.5)
  points(x, policy, pch = pch4, col = col4, cex = 1.5)
  
  arrows(x_oracle, oracle - oracle_se, x_oracle, pmin(oracle + oracle_se, 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col1)
  arrows(x, plugin - plugin_se, x, pmin(plugin + plugin_se, 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col2)
  arrows(x_eif_oracle, eif_oracle - eif_oracle_se, x_eif_oracle, pmin(eif_oracle + eif_oracle_se, 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col3)
  arrows(x, policy - policy_se, x, pmin(policy + policy_se, 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col4)
  
  segments(min(x), min(x), max(x), max(x), lty = 2, lwd = 2)
  if (Gamma < 1) segments(Gamma, 0.4, Gamma, 1, lty = 3, lwd = 2)
  
  if (show_legend) legend("bottomright", legend = c("Semi-oracle", "Plug-in", "EIF-oracle", "Policy"), col = c(col1, col2, col3, col4), pch = c(pch1, pch2, pch3, pch4), bty = "n", cex = 1.8)
}

par(mfrow = c(2, 3), mgp = c(3.5, 1, 0), cex.lab = 2.2, cex.axis = 2, mar = c(2.5, 5, 4, 0), oma = c(2.5, 1, 0, 2), xpd = NA, font.main = 1, cex.main = 2)

plot_policy(res11, 1, title = TRUE, ylab = "Training size = 2000", show_legend = TRUE)
plot_policy(res12, 0.9, title = TRUE)
plot_policy(res13, 0.8, title = TRUE)

plot_policy(res21, 1, ylab = "Training size = 500", xlab = expression(1-alpha))
plot_policy(res22, 0.9, xlab = expression(1-alpha))
plot_policy(res23, 0.8, xlab = expression(1-alpha))






res11 <- read.table("policy_Gamma_1_setting_1_tr_2000_new.txt", header = TRUE)
res12 <- read.table("policy_Gamma_09_setting_1_tr_2000_new.txt", header = TRUE)
res13 <- read.table("policy_Gamma_08_setting_1_tr_2000_new.txt", header = TRUE)
res21 <- read.table("policy_Gamma_1_setting_1_tr_500_new.txt", header = TRUE)
res22 <- read.table("policy_Gamma_09_setting_1_tr_500_new.txt", header = TRUE)
res23 <- read.table("policy_Gamma_08_setting_1_tr_500_new.txt", header = TRUE)

alpha_set <- seq(0.05, 0.30, by = 0.05)

col1 <- "#5A5A5A"
col2 <- "#E9A95B"
col3 <- "#5B9BD5"
col4 <- "#D9796F"

pch1 <- 16
pch2 <- 15
pch3 <- 17
pch4 <- 18

plot_policy <- function(res, Gamma, title = FALSE, ylab = "", xlab = "", show_legend = FALSE) {
  x <- 1 - alpha_set
  offset <- 0.005
  x_oracle <- x - offset
  x_eif_oracle <- x + offset
  
  oracle <- res$true_oracle_coverage_mean
  plugin <- res$plugin_coverage_mean
  eif_oracle <- res$eif_oracle_policy_coverage_mean
  policy <- res$eif_estimated_coverage_mean
  
  oracle_se <- res$true_oracle_coverage_se
  plugin_se <- res$plugin_coverage_se
  eif_oracle_se <- res$eif_oracle_policy_coverage_se
  policy_se <- res$eif_estimated_coverage_se
  
  matplot(x, cbind(oracle, plugin, eif_oracle, policy), type = "n", xlab = xlab, ylab = ylab, ylim = c(0.7, 1), main = if (title) bquote(Gamma == .(Gamma)) else "")
  
  points(x_oracle, oracle, pch = pch1, col = col1, cex=1.5)
  points(x, plugin, pch = pch2, col = col2, cex=1.5)
  points(x_eif_oracle, eif_oracle, pch = pch3, col = col3, cex=1.5)
  points(x, policy, pch = pch4, col = col4, cex=1.5)
  
  arrows(x_oracle, oracle - oracle_se, x_oracle, pmin(oracle + oracle_se, 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col1)
  arrows(x, plugin - plugin_se, x, pmin(plugin + plugin_se, 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col2)
  arrows(x_eif_oracle, eif_oracle - eif_oracle_se, x_eif_oracle, pmin(eif_oracle + eif_oracle_se, 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col3)
  arrows(x, policy - policy_se, x, pmin(policy + policy_se, 1), length = 0.01, angle = 90, code = 3, lwd = 2, col = col4)
  
  segments(min(x), min(x), max(x), max(x), lty = 2, lwd = 2)
  if (Gamma < 1) segments(Gamma, 0.7, Gamma, 1, lty = 3, lwd = 2)
  
  if (show_legend) legend("bottomright", legend = c("Semi-oracle", "Plug-in", "EIF-oracle", "Policy"), col = c(col1, col2, col3, col4), pch = c(pch1, pch2, pch3, pch4), bty = "n", cex = 1.8)
}

par(mfrow = c(2, 3), mgp = c(3.5, 1, 0), cex.lab = 2.2, cex.axis = 2, mar = c(2.5, 5, 4, 0), oma = c(2.5, 1, 0, 2), xpd = NA, font.main = 1, cex.main = 2)

plot_policy(res11, 1, title = TRUE, ylab = "Training size = 2000", show_legend = TRUE)
plot_policy(res12, 0.9, title = TRUE)
plot_policy(res13, 0.8, title = TRUE)

plot_policy(res21, 1, ylab = "Training size = 500", xlab = expression(1-alpha))
plot_policy(res22, 0.9, xlab = expression(1-alpha))
plot_policy(res23, 0.8, xlab = expression(1-alpha))



##known p_{Z|X}

res11 <- read.table("known_pzx_K_10_ntrain_2000.txt", header = TRUE)
res12 <- read.table("known_pzx_K_10_ntrain_10000.txt", header = TRUE)
res21 <- read.table("known_pzx_K_2_ntrain_2000.txt", header = TRUE)
res22 <- read.table("known_pzx_K_2_ntrain_10000.txt", header = TRUE)

col1 <- "#D95F5F"
col2 <- "#4F81BD"

par(mfrow = c(2, 2), mgp = c(3.5, 1, 0), cex.lab = 2, cex.axis = 2, mar = c(2.5, 5, 4, 1), oma = c(2.5, 1, 0, 1), xpd = NA, font.main = 1, cex.main = 2)

matplot(1 - alpha_set, res11[, c("WCP_coverage", "EIF_coverage")], type = "b", pch = c(16, 17), lty = 1, lwd = 2, col = c(col1, col2), cex = 2, xlab = "", ylab = "Setting 1", ylim = c(0.6, 1), main = "Training size = 2000")
segments(min(1 - alpha_set), min(1 - alpha_set), max(1 - alpha_set), max(1 - alpha_set), lty = 2, lwd = 2)
legend("bottomright", c("WCP", "EIF-based"), col = c(col1, col2), pch = c(16, 17), lty = 1, lwd = 2, pt.cex = 2, bty = "n", cex = 1.5)
arrows(1-alpha_set, res11$WCP_coverage-res11$WCP_se, 1-alpha_set, pmin(res11$WCP_coverage+res11$WCP_se,1), length=0.01, angle=90, code=3, lwd=2, col=col1)
arrows(1-alpha_set, res11$EIF_coverage-res11$EIF_se, 1-alpha_set, pmin(res11$EIF_coverage+res11$EIF_se,1), length=0.01, angle=90, code=3, lwd=2, col=col2)

matplot(1 - alpha_set, res12[, c("WCP_coverage", "EIF_coverage")], type = "b", pch = c(16, 17), lty = 1, lwd = 2, col = c(col1, col2), cex = 2, xlab = "", ylab = "", ylim = c(0.6, 1), main = "Training size = 10000")
segments(min(1 - alpha_set), min(1 - alpha_set), max(1 - alpha_set), max(1 - alpha_set), lty = 2, lwd = 2)
arrows(1-alpha_set, res12$WCP_coverage-res12$WCP_se, 1-alpha_set, pmin(res12$WCP_coverage+res12$WCP_se,1), length=0.01, angle=90, code=3, lwd=2, col=col1)
arrows(1-alpha_set, res12$EIF_coverage-res12$EIF_se, 1-alpha_set, pmin(res12$EIF_coverage+res12$EIF_se,1), length=0.01, angle=90, code=3, lwd=2, col=col2)

matplot(1 - alpha_set, res21[, c("WCP_coverage", "EIF_coverage")], type = "b", pch = c(16, 17), lty = 1, lwd = 2, col = c(col1, col2), cex = 2, xlab = expression(1 - alpha), ylab = "Settin 2", ylim = c(0.6, 1))
segments(min(1 - alpha_set), min(1 - alpha_set), max(1 - alpha_set), max(1 - alpha_set), lty = 2, lwd = 2)
arrows(1-alpha_set, res21$WCP_coverage-res21$WCP_se, 1-alpha_set, pmin(res21$WCP_coverage+res21$WCP_se,1), length=0.01, angle=90, code=3, lwd=2, col=col1)
arrows(1-alpha_set, res21$EIF_coverage-res21$EIF_se, 1-alpha_set, pmin(res21$EIF_coverage+res21$EIF_se,1), length=0.01, angle=90, code=3, lwd=2, col=col2)

matplot(1 - alpha_set, res22[, c("WCP_coverage", "EIF_coverage")], type = "b", pch = c(16, 17), lty = 1, lwd = 2, col = c(col1, col2), cex = 2, xlab = expression(1 - alpha), ylab = "", ylim = c(0.6, 1))
segments(min(1 - alpha_set), min(1 - alpha_set), max(1 - alpha_set), max(1 - alpha_set), lty = 2, lwd = 2)
arrows(1-alpha_set, res22$WCP_coverage-res22$WCP_se, 1-alpha_set, pmin(res22$WCP_coverage+res22$WCP_se,1), length=0.01, angle=90, code=3, lwd=2, col=col1)
arrows(1-alpha_set, res22$EIF_coverage-res22$EIF_se, 1-alpha_set, pmin(res22$EIF_coverage+res22$EIF_se,1), length=0.01, angle=90, code=3, lwd=2, col=col2)




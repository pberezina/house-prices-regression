
# load some useful packages
suppressPackageStartupMessages(require(lars, quietly=TRUE))
suppressPackageStartupMessages(require(ggplot2, quietly=TRUE))
suppressPackageStartupMessages(require(gglasso, quietly=TRUE))
suppressPackageStartupMessages(require(dplyr, quietly=TRUE))
suppressPackageStartupMessages(require(MASS, quietly=TRUE))
suppressPackageStartupMessages(require(gridExtra, quietly=TRUE))
suppressPackageStartupMessages(require(glmnet, quietly=TRUE))
suppressPackageStartupMessages(require(car, quietly=TRUE))

# set seed for reproducibility
set.seed(23984)

#  load in the dataset
all.dat <- read.table("datasets/pruned_data.csv", sep=",", header=TRUE)
all.dat <- all.dat[,which(colnames(all.dat)!="Id")]

# get log transformed Sale Price Dataset
log.dat <- all.dat
log.dat$SalePrice <- sqrt(all.dat$SalePrice)

# get trimmed Sale Price Dataset (under 4e5)
tr.dat <- all.dat[which(all.dat$SalePrice < 4.0*10^5),]

# get trimmed log transformed Sale Price Dataset
trlg.dat <- tr.dat
trlg.dat$SalePrice <- sqrt(tr.dat$SalePrice)


par(mfrow=c(1,4))
hist(all.dat$SalePrice, col="Slategray1", main="Hist. of SalePrice", xlab="SalePrice")
hist(log.dat$SalePrice, col="Slategray1", main="Hist. of log10(SalePrice)", xlab="log10(SalePrice)")
hist(tr.dat$SalePrice, col="Slategray1", main="Hist. of trunc. SalePrice", xlab="trunc. SalePrice")
hist(trlg.dat$SalePrice, col="Slategray1", main="Hist. of log10(trunc. SalePrice)", xlab="log10(trunc. SalePrice)")
par(mfrow=c(1,1))

# define predictor and outcome for each set
## all.dat untransformed
X.all <- model.matrix(SalePrice~., data=all.dat)
Y.all <- all.dat[,"SalePrice"]

## log.dat transformed
X.log <- model.matrix(SalePrice~., data=log.dat)
Y.log <- log.dat[,"SalePrice"]

## tr.dat truncated data
X.tr <- model.matrix(SalePrice~., data=tr.dat)
Y.tr <- tr.dat[,"SalePrice"]

## trlg.dat truncated log trasformed data
X.trlg <- model.matrix(SalePrice~., data=trlg.dat)
Y.trlg <- trlg.dat[,"SalePrice"]

################################################3
### Lasso Regressions

runLASSO <- function(X,Y,title) {
  
  # run the lasso
  fit <- cv.glmnet(x=X,y=Y, family="gaussian", type.measure="mse", alpha=1)
  
  # plot the outputs
  png(sprintf("%s_panel.png",title), height=394, width=1391)
  par(mfrow=c(1,5))
  p1 <- plot(fit, main=sprintf("Log Lambda vs MSE\n",title))
  p2 <- plot(fit$glmnet.fit, main=sprintf("Coefficient Shrinkage\n",title))
  
  # gather some important internal attributes
  lam <- fit$lambda.min
  fitted.vals <-predict(fit, s=lam, newx = X,type="response")
  resid.vals <- (Y - fitted.vals)
  SST <- sum((Y - mean(Y))^2)
  SSE <- sum((Y - fitted.vals)^2)
  SSR <- SST - SSE
  n <- nrow(X)
  p <- length(coef(fit)[which(coef(fit)!=0)])
  dfe <- n-p-1
  dft <- n-1

  # calculate R^2 and adjusted R^2
  Rsq <- 1-SSE/SST
  adj.Rsq <- 1 - ((SSE/dfe) / (SST/dft))
  
  # print R^2 and adjusted R^2
  print(sprintf("Model Rsq: %0.4f", Rsq))
  print(sprintf("Model adj. Rsq: %0.4f", adj.Rsq))
  
  # plot the predicted vs outcome figure
  p3 <- plot(x=fitted.vals, y=Y, xlab="Predicted",
       ylab="True Y", main=sprintf("Predictions vs Actual\n (Adj. Rsq: %0.3f)", adj.Rsq),
       pch=19, col="black")
  p3 <- p3 + abline(0,1, col="darkred", lwd=3)
  
  # plot residuals vs fitted values
  p4 <- plot(x=fitted.vals, y=resid.vals,
       pch=19, main="Residuals vs Fitted",
       xlab="Fitted Values",ylab="Residuals")
  p4 <- p4 + abline(h=0, lwd=3, col="darkred")
  
  # plot QQ plot
  p5 <- qqPlot(resid.vals, main = "QQ-plot")
  dev.off()
  
  # print the coefficient list
  print(coef(fit, s="lambda.min"))
  print(row.names(coef(fit, s="lambda.min"))[which(coef(fit,s="lambda.min")!=0)])
  print(rownames(coef(fit, s="lambda.min"))[sort(coef(fit,s="lambda.min"), decreasing = T)])
  
  return(fit)
}


# run the lasso function we just built
runLASSO(X=X.all, Y=Y.all, title="Untransformed Response")
runLASSO(X=X.log, Y=Y.log, title="Root Transform Response")
runLASSO(X=X.tr, Y=Y.tr, title="Truncated Response")
runLASSO(X=X.trlg, Y=Y.trlg, title="root Tr. Truncated Response")




# define group lasso protocol
runGRPLASSO <- function(X,Y, grp, title) {
  
  # run the lasso
  fit <- cv.gglasso(x=X,y=Y, group=grp, pred.loss="L1", nfolds=10, lambda.factor=0.01)
  
  # plot the outputs
  png(sprintf("%s_panel.png",title), height=394, width=1391)
  par(mfrow=c(1,5))
  p1 <- plot(fit, main=sprintf("Log Lambda vs MSE\n",title))
  p2 <- plot(fit$gglasso.fit, main=sprintf("Coefficient Shrinkage\n",title))
  
  # gather some important internal attributes
  lam <- fit$lambda.min
  fitted.vals <-predict.gglasso(fit$gglasso.fit, s=lam, newx = X,type="link")
  resid.vals <- (Y - fitted.vals)
  SST <- sum((Y - mean(Y))^2)
  SSE <- sum((Y - fitted.vals)^2)
  SSR <- SST - SSE
  n <- nrow(X)
  p <- length(coef(fit)[which(coef(fit)!=0)])
  dfe <- n-p-1
  dft <- n-1
  
  # calculate R^2 and adjusted R^2
  Rsq <- 1-SSE/SST
  adj.Rsq <- 1 - ((SSE/dfe) / (SST/dft))
  
  # print R^2 and adjusted R^2
  print(sprintf("Model Rsq: %0.4f", Rsq))
  print(sprintf("Model adj. Rsq: %0.4f", adj.Rsq))
  
  # plot the predicted vs outcome figure
  p3 <- plot(x=fitted.vals, y=Y, xlab="Predicted",
             ylab="True Y", main=sprintf("Predictions vs Actual\n (Adj. Rsq: %0.3f)", adj.Rsq),
             pch=19, col="black")
  p3 <- p3 + abline(0,1, col="darkred", lwd=3)
  
  # plot residuals vs fitted values
  p4 <- plot(x=fitted.vals, y=resid.vals,
             pch=19, main="Residuals vs Fitted",
             xlab="Fitted Values",ylab="Residuals")
  p4 <- p4 + abline(h=0, lwd=3, col="darkred")
  
  # plot QQ plot
  p5 <- qqPlot(resid.vals, main = "QQ-plot")
  dev.off()
  
}

## group LASSO 
## get groups 
grps <- c(1)
colnum <- 2
for(c in colnames(all.dat[,which(colnames(all.dat)!="SalePrice")])) {
  # detect if it is a factor or numeric
  if(is.factor(all.dat[,c])) {
    #  if it is a factor get how many levels there are
    # and append to the groups 
    nlevels <- length(levels(all.dat[,c]))
    grps <- append(grps, rep(colnum, nlevels-1))
    colnum <- colnum + 1
  } else {
    grps <- append(grps, colnum)
    colnum <- colnum + 1
  }
}


runGRPLASSO(X=X.all, Y=Y.all, grp=grps, title="All Data Group")
runGRPLASSO(X=X.log, Y=Y.log, grp=grps, title="root tr. Group")
runGRPLASSO(X=X.tr, Y=Y.tr, grp=grps, title="tunc. Group")
runGRPLASSO(X=X.trlg, Y=Y.trlg, grp=grps, title="root tr trunc. Group")


library(survival)
nsim <-100
set.seed(628)
library(mvtnorm)
for(w in 1:nsim) {
  sigma <-2.5
  s<-200#number of cluster
  c_s<-sample(20:100, s,replace = T)#cluster size
  n<-sum(c_s)
  b<-x1<-x2<-x3<-x4<-x5<-x6<-rep(NA,n)
  
  for(i in 1:s){
    for(j in 1:c_s[i]){
      if(i==1){
        b[1:c_s[1]]<-rep(rnorm(1,0,sqrt(sigma)),c_s[1])
      }else{
        b[sum(c_s[1:(i-1)],1):sum(c_s[1:(i-1)],c_s[i])]<-rep(rnorm(1,0,sqrt(sigma)),c_s[i])
      }
    }
  }
  rho=0
  Sigma<-matrix(rho,5,5)
  diag(Sigma)<-rep(1,5)
  metrx<-rmvnorm(n,rep(0,5),
                 sigma= Sigma)
  x1<-metrx[,1]
  x2<-metrx[,2]
  x3<-metrx[,3]
  x4<-metrx[,4]
  x5<-metrx[,5]

  nx1<-x1^2
  nx2<-x2^2
  nx3<-x3^2
  nx4<-x4^2
  nx5<-x5^2
  u<-runif(n,0,1)
  t<-rep(NA,n)
  t<-unlist(lapply(1:n,function(x){
    t[x]<-exp((-sum(c(nx1[x],nx2[x],nx3[x],nx4[x],nx5[x]))/2-b[x]+3))*log(1/(1-u[x]))
  }))
  
  
 
  tau<-0.5
  c<-runif(n,0,tau)
  delta<-as.numeric(t<=c)
  o<-delta*t+(1-delta)*c
  cr[w]<-mean(1-delta)
  
  Subject<-unlist(lapply(1:s,function(x){rep(x,c_s[x])}))
  #z<-rep(1,n)
  data1<-data.frame(cbind(o,delta,x1,x2,x3,x4,x5,nx1,nx2,nx3,nx4,nx5,Subject))
  data1<-data1[order(data1$Subject),]
  data1$u<-data1$o
  data1$status<-data1$delta
  split<-list()
  for(i in 1:length(unique(data1$Subject)) ){
    split[[i]]<-c(rep(0,round(c_s[i]*0.5, digits = 0)),rep(1,(c_s[i]-round(c_s[i]*0.5, digits = 0))))
    
  }
 
  data1$split_train<-unlist(split)
  
  re<-matrix(0,nrow=n,ncol=s)
  
  for(i in 1:s){
    
    if(i==1){
      re[1:c_s[1]]<-rep(1,c_s[1])
    }else{
      re[sum(c_s[1:(i-1)],1):sum(c_s[1:(i-1)],c_s[i]),i]<-rep(1,c_s[i])
    }
    
  }
  sum(apply(re,1,sum)>1)
  data1_nn<-data1[c("o","delta","x1","x2","x3","x4","x5","split_train")]
  
  data1_nn.train<-data1_nn[data1_nn$split_train==0,]
  data1_nn.val<-data1_nn[data1_nn$split_train==0,]
  data1_nn.test<-data1_nn[data1_nn$split_train==1,]
  
  
  data1.train<-data1[data1_nn$split_train==0,]
  data1.val<-data1[data1_nn$split_train==0,]
  data1.test<-data1[data1_nn$split_train==1,]
  re.train<-re[data1_nn$split_train==0,]
  re.val<-re[data1_nn$split_train==0,]
  re.test<-re[data1_nn$split_train==1,]
  write.csv(data1_nn.train,file=paste("user/var2_5/train/datatrain",w,".csv",sep=""))
  write.csv(data1_nn.val,file=paste("user/var2_5/val/dataval",w,".csv",sep=""))
  write.csv(data1_nn.test,file=paste("user/var2_5/test/datatest",w,".csv",sep=""))
  write.csv(re.train,file=paste("user/var2_5/re_train/retrain",w,".csv",sep=""))
  write.csv(re.val,file=paste("user/var2_5/re_val/reval",w,".csv",sep=""))
  write.csv(re.test,file=paste("user/var2_5/re_test/retest",w,".csv",sep=""))

}

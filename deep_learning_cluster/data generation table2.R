library(survival)
library(mvtnorm)
library(Matrix)
nsim <-100

for(w in 1:nsim) {
  
  
  sigma <-2.5##variance 
  s<-200#number of cluster
  c_s<-sample(20:100, s,replace = T)#cluster size
  n<-sum(c_s)
  b<-rep(NA,n)
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
  
  Sigma<-matrix(rho,15,15)
  diag(Sigma)<-rep(1,15)
  metrx<-rmvnorm(n,c(1,1,1,2,2,3,3,3,0,0,0,0,0,0,0),sigma= Sigma)
  x1<-metrx[,1]
  x2<-metrx[,2]
  x3<-metrx[,3]
  x4<-metrx[,4]
  x5<-metrx[,5]
  x6<-metrx[,6]
  x7<-metrx[,7]
  x8<-as.numeric(metrx[,8]<1)
  x9<-metrx[,9]
  x10<-metrx[,10]
  x11<-metrx[,11]
  x12<-metrx[,12]
  x13<-metrx[,13]
  x14<-metrx[,14]
  x15<-metrx[,15]
  nx1<-exp(x1*(1+x2-x3*x4*x5)/2)*abs(x5+0.2-0.01*(x9*x10))/10
  nx2<-x5*(x4*x3-0.3)/(abs(2*x4*x6*x3-1+0.01*(x11*x12))+1)
  nx3<-2*sin(x5*x2*x1)*abs(x5*x6*x2-0.6-(0.01*(x13*x14)))
  nx4<-log(abs(x6*x1*x2)+abs(x5*x7*x8+0.01*x15))
  nx1<-(nx1-mean(nx1))/sd(nx1)
  nx2<-(nx2-mean(nx2))/sd(nx2)
  nx3<-(nx3-mean(nx3))/sd(nx3)
  nx4<-(nx4-mean(nx4))/sd(nx4)
  u<-runif(n,0,1)
  t<-rep(NA,n)
  t<-unlist(lapply(1:n,function(x){
    t[x]<-exp((-sum(c(nx1[x],nx2[x],nx3[x],nx4[x]))+4-b[x]))*log(1/(1-u[x]))
  }))
  
  summary(t)
  tau<-30
  c<-runif(n,0,tau)
  delta<-as.numeric(t<=c)
  o<-delta*t+(1-delta)*c
  cr[w]<-mean(1-delta)
  
  
  Subject<-unlist(lapply(1:s,function(x){rep(x,c_s[x])}))
  data1<-data.frame(cbind(o,delta,x1,x2,x3,x4,x5,
                          x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,nx1,nx2,nx3,nx4,Subject))
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
    # len<-dim(kidney[kidney$facility== unique(kidney$facility)[i+1],])[1]
    
  }
  sum(apply(re,1,sum)>1)
  data1_nn<-data1[c("o","delta","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15",
                    "split_train")]
  
  data1_nn.train<-data1_nn[data1_nn$split_train==0,]
  data1_nn.val<-data1_nn[data1_nn$split_train==0,]
  data1_nn.test<-data1_nn[data1_nn$split_train==1,]
  
  
  data1.train<-data1[data1_nn$split_train==0,]
  data1.val<-data1[data1_nn$split_train==0,]
  data1.test<-data1[data1_nn$split_train==1,]
  re.train<-re[data1_nn$split_train==0,]
  re.val<-re[data1_nn$split_train==0,]
  re.test<-re[data1_nn$split_train==1,]
  #  
  #  
  write.csv(data1_nn.train,file=paste("user/var0/train/datatrain",w,".csv",sep=""))
  write.csv(data1_nn.val,file=paste("user/var0/val/dataval",w,".csv",sep=""))
  write.csv(data1_nn.test,file=paste("user/var0/test/datatest",w,".csv",sep=""))
  write.csv(re.train,file=paste("user/var0/re_train/retrain",w,".csv",sep=""))
  write.csv(re.val,file=paste("user/var0/re_val/reval",w,".csv",sep=""))
  write.csv(re.test,file=paste("user/var0/re_test/retest",w,".csv",sep=""))
  
 
}


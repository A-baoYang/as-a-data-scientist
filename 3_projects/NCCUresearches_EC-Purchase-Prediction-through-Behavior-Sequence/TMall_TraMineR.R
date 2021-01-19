#install.packages("factoextra")
#install.packages("NbClust")
#install.packages("viridis")
#install.packages("RColorBrewer")
library("TraMineR")
library("cluster")
library("factoextra")
library("NbClust")
library('caret')
library('RcmdrMisc')
library('ggplot2')
#library("viridis")
#library("RColorBrewer")
tmall <- read.csv('order_logistic_states-20201215-v6.csv')
### Sequence Formatting
tmall.seq <- seqdef(tmall, 3:128, xtstep=10)
tmall.seq.cost <- seqcost(tmall.seq, method="INDELSLOG")
# seqmtplot(tmall.seq, group = mvad$funemp, ylim=c(0,30))
# by(tmall.seq, mvad$funemp, seqmeant)

### Transition rates between states
tmall.trate <- round(seqtrate(tmall.seq, time.varying = FALSE), 2)

### pairwise optimal matching distances between sequences
#### OM
tmall.om <- seqdist(tmall.seq, method = 'OM', sm = 'TRATE', indel = 'auto')

#### OMloc
tmall.om <- seqdist(tmall.seq, method = 'OMloc', sm = 'TRATE', indel = 'auto')

#### OMslen
tmall.om <- seqdist(tmall.seq, method = 'OMslen', sm = 'TRATE', indel = 'auto')

#### OMspell
tmall.om <- seqdist(tmall.seq, method = 'OMspell', sm = 'TRATE', indel = "auto")

#### OMstran
tmall.om <- seqdist(tmall.seq, method = 'OMstran', sm = 'TRATE', indel = "auto", otto = 0.5, with.missing = TRUE)

#### CHI
tmall.om <- seqdist(tmall.seq, method = 'CHI2')

#### SVR
tmall.om <- seqdist(tmall.seq, method = 'SVRspell')

#### NMS
tmall.om <- seqdist(tmall.seq, method = 'NMS')
tmall.om <- seqdist(tmall.seq, method = 'NMSMST')

#### others
tmall.om <- seqdist(tmall.seq, method = 'TWED', nu = 0.5, sm = 'INDELSLOG')


### Clustering
#### finding optimal number of clusters
fviz_nbclust(tmall.om, FUNcluster = hcut, method = "silhouette", k.max = 12, print.summary = TRUE)
# https://rpubs.com/skydome20/R-Note9-Clustering
# wss = within sum of square, silhouette = avg. silhouette method

#### clustering
#clusterward <- agnes(tmall.seq, diss = FALSE, method = 'ward')  # error
clusterward <- agnes(tmall.om, diss = TRUE, method = 'ward')  # "diss" determine the x object type
tmall.cl4 <- cutree(clusterward, k=2)
tmall.cl4 <- kmeans(tmall.om, centers = 2)$cluster  # https://rpubs.com/skydome20/R-Note9-Clustering
cl4.lab <- factor(tmall.cl4, labels = paste("Cluster"), 1:2)  # change clust number to factor(text)

### plots
colorset.v3.1 <- c("gray", "forestgreen", "dodgerblue", "darkorange3", 
                   "yellowgreen", "skyblue", "gold", "white")
colorset.v3.2 <- c("gray", "forestgreen", "darkorange3", 
                   "green", "gold", "white")
colorset.v3.3 <- c("gray", "forestgreen", "dodgerblue", "darkorange3", 
               "yellowgreen", "skyblue", "gold", "white")
colorset.v3.4 <- c("white","aquamarine3","azure1","azure2","aquamarine1","azure3","aquamarine4","darkorange1","darkorange2")

seqdplot(tmall.seq, group = cl4.lab, border=NA, cpal=colorset.v3.4)
seqfplot(tmall.seq, group = cl4.lab, border=NA)
seqiplot(tmall.seq, group = cl4.lab, border=NA)
seqIplot(tmall.seq, group = cl4.lab, border=NA)
seqHtplot(tmall.seq, group = cl4.lab, border=NA)  # transversal entropy from seqstatd
seqmsplot(tmall.seq, group = cl4.lab, border=NA)  # modal state sequence from seqmodst
seqmtplot(tmall.seq, group = cl4.lab, border=NA)  # mean time spent from seqmeant
seqrplot(tmall.seq, group = cl4.lab, border=NA)  # representative sequence
seqpcplot(tmall.seq, group = cl4.lab, border=NA, order.align = "time")  # decorated parallel coordinate, sequences are displayed as jittered frequency-weighted parallel lines
seqiplot(tmall.seq, border=NA, withlegend='right')


###
# entropies <- seqient(tmall.seq)
# lm.ent <- lm(entropies ~ male + funemp + gcse5eq, mvad)

### Loop with cluster_nums
target_num=c()
avg_percent=c()
target_num[1] <- 0
avg_percent[1] <- 0
for (k in 2:50) {
  result <- k_loop(tmall.om, cluster.method = 'kmeans', cluster.k=k, threshold=0.6)
  target_num[k] <- result[1]
  avg_percent[k] <- result[2]
}
plot_records(target_num, avg_percent)

### Plot - https://stackoverflow.com/questions/6142944/how-can-i-plot-with-2-different-y-axes
plot_records <- function(target_num, avg_percent) {
  plot(target_num, pch=16, axes=FALSE, ylim=c(0,max(target_num)), xlab="", ylab="", 
       type="o",col="red", main="target_num")
  axis(2, ylim=c(0,max(target_num)), col="red", las=1)
  mtext("target_num", side=2, line=2.5)
  abline(a=100, b=0, col="pink")
  box()
  par(new=TRUE)
  ## Plot the second plot and put axis scale on right
  plot(avg_percent, pch=15,  xlab="", ylab="", ylim=c(0,1), 
       axes=FALSE, type="b", col="blue")
  ## a little farther out (line=4) to make room for labels
  axis(4, ylim=c(0,1), col="blue", col.axis="blue", las=1)
  mtext("avg_percent", side=4, col="blue", line=4)
  ## Draw the time axis
  axis(1, pretty(seq(1,length(target_num),1), 10))
  mtext("Clusters", side=1, col="black", line=2.5)
  ## Add Legend
  legend("topleft", legend=c("target_num", "avg_percent"),
         text.col=c("red","blue"), pch=c(16,15), col=c("red","blue"))
}

### Paramter test functions ###
k_loop <- function(tmall.om, cluster.method, cluster.metric='euclidean', cluster.k, threshold) {
  if (cluster.method=='kmeans') {
    tmall.cl4 <- kmeans(tmall.om, centers = cluster.k)$cluster
  } else {
    clusterward <- agnes(tmall.om, diss = TRUE, metric = cluster.metric, method = cluster.method)  # "diss" determine the x object type
    tmall.cl4 <- cutree(clusterward, k=cluster.k)
    if (cluster.k>1) {
      cl4.lab <- factor(tmall.cl4, labels = paste("Cluster"), 1:cluster.k)  # change clust number to factor(text)
    } else {
      cl4.lab <- factor(tmall.cl4, labels = paste("Cluster"), 1)  # change clust number to factor(text)
    }
  }
  
  # False Classification
  ###flag back to dataframe
  tmall$omcluster <- tmall.cl4
  ###count confusion matrix
  tmall$omcluster <- as.factor(tmall$omcluster)
  tmall$review_score <- as.factor(tmall$review_score)
  # confusionMatrix(tmall$omcluster, tmall$flag, positive = '1')
  composition = table(tmall$omcluster, tmall$review_score)
  composition <- cbind(composition, round(composition[,'1'] / (composition[,'0']+composition[,'1']), 2))
  colnames(composition) <- c('label.0','label.1','percentage')
  composition <- data.frame(composition)
  composition <- composition[order(composition$percentage, decreasing = TRUE), ]
  composition <- composition[composition$percentage>threshold, ]
  
  cat('cluster method: ', cluster.method, '\n')
  cat('cluster metric: ', cluster.metric, '\n')
  cat('cluster num: ', cluster.k, '\n')
  # print(composition)
  print(composition)
  cat('target audience: ', sum(composition$label.1), '\n')
  cat('total audience: ', sum(composition$label.0)+sum(composition$label.1), '\n')
  cat('average percentage: ', sum(composition$label.1) / (sum(composition$label.0)+sum(composition$label.1)), '\n')
  cat('===========', '\n')
  
  return(c(sum(composition$label.1), sum(composition$label.1) / (sum(composition$label.0)+sum(composition$label.1))))
}


omcluster <- function(tmall.seq, method, sm, indel='auto', otto=0.5, nu=0.9, cluster.method, cluster.metric='euclidean', cluster.k){
  if (method == 'OM' || method == 'OMspell' || method == 'OMloc' || method == 'OMslen') {
    tmall.om <- seqdist(tmall.seq, method = method, sm = sm, indel = indel)
  } else if (method == 'OMstran') {
    tmall.om <- seqdist(tmall.seq, method = method, sm = sm, indel = indel, otto = otto)
  } else if (method == 'TWED') {
    tmall.om <- seqdist(tmall.seq, method = method, nu = otto, sm = sm)
  } else {
    tmall.om <- seqdist(tmall.seq, method = method)
  }
  cat('OM method: ', method, '\n')
  cat('SM method: ', sm, '\n')
  
  k_loop(tmall.om, cluster.method, cluster.metric='euclidean', cluster.k)
  
}


###OM Family & TWED
om.methods = c('OM','OMspell','OMslen','OMloc','OMstran','TWED')
cluster.metrics = c('euclidean', 'manhattan')
cluster.methods = c('gaverage', 'ward')
sm.methods = c('INDELS','INDELSLOG','TRATE')

tryCatch({
  for (o in 1:length(om.methods)) {
    for (s in 1:length(sm.methods)) {
      for (c in 1:length(cluster.methods)) {
        for (m in 1:length(cluster.metrics)) {
          for (cluster.k in 2:10) {
            if (om.methods[o] == 'OM' || om.methods[o] == 'OMspell' || om.methods[o] == 'OMslen' || om.methods[o] == 'OMloc') {
              omcluster(tmall.seq, method=om.methods[o], sm=sm.methods[s], indel='auto', cluster.method=cluster.methods[c], cluster.metric=cluster.metrics[m], cluster.k=cluster.k)
            } else if (om.methods[o] == 'OMstran') {
              omcluster(tmall.seq, method=om.methods[o], sm=sm.methods[s], indel='auto', otto=0.5, cluster.method=cluster.methods[c], cluster.metric=cluster.metrics[m], cluster.k=cluster.k)
            } else {
              omcluster(tmall.seq, method=om.methods[o], sm=sm.methods[s], indel='auto', nu=0.9, cluster.method=cluster.methods[c], cluster.metric=cluster.metrics[m], cluster.k=cluster.k)
            }
          }
        }
      }
    }
  }
}, warning = function() {
  cat(om.methods[o], sm.methods[s], cluster.methods[c], cluster.k)
  warning('some parmas might not used or should be used.')
})


## non OM Family
seqdist.methods = c('CHI2', 'SVRspell', 'NMS', 'NMSMST')
cluster.methods = c('gaverage', 'ward')
tryCatch({
  for (o in 1:length(seqdist.methods)) {
      for (c in 1:length(cluster.methods)) {
          for (cluster.k in 2:10) {
            omcluster(tmall.seq, method=seqdist.methods[o], sm='TRATE', indel='auto', cluster.method=cluster.methods[c], cluster.metric=cluster.metrics[m], cluster.k=cluster.k)
          }
      }
  }
}, warning = function() {
  cat(om.methods[o], sm.methods[s], cluster.methods[c], cluster.k)
  warning('some parmas might not used or should be used.')
})

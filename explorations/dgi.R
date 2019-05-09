X.train.dgi <- read.csv("~/Downloads/X_train.csv", header = FALSE)
y.train.dgi <- read.csv("~/Downloads/y_train.csv", header = FALSE)

set.seed(2356789)
data.dgi <- data.frame(X.train.dgi, y = y.train.dgi[, 1])
dgi.rf = ranger(y~., data = data.dgi, 
       num.trees = 100,
       mtry = 3,
       importance = "impurity",
       max.depth = 20)
imp_org = dgi.rf$variable.importance
imp_org = imp_org/sum(imp_org)

dgi.perm = ranger(y~., data = data.dgi, 
                num.trees = 100,
                mtry = 3,
                importance = "impurity_corrected",
                max.depth = 20)
imp_ranger = dgi.perm$variable.importance
imp_ranger = imp_ranger/sum(imp_ranger)

dgi.cforest = cforest(y~., data = data.dgi,
                           control = cforest_control(ntree = 100, 
                                                     mtry = 3, 
                                                     maxdepth = 20))
imp_cforest = varimp(dgi.cforest)
imp_cforest = imp_cforest/sum(imp_cforest)

imp = data.frame(imp_org, imp_ranger, imp_cforest)
write.csv(imp, "imp.csv")



m=load('start.mat');
par=[0.3,1,0.1];% epsilon,gamma,alpha

[cumr,Q]=RLearning(m.Rew_strCliff,m.transitionCliff,par);
c=conv(cumr,2*normpdf (1:100,100,30));plot(c(100:1000))


par=[0,1,0.1];
[cumr,Q]=RLearning(m.Rew_strCliff,Q,par);

SCtrain=sum(cumr)
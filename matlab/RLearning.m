function [cumr,Q]=RLearning(Rew_str, transition,par,plotting)
%Written by Ulrik Beierholm May 2008, modified Feb 2012
%A simple implementation of grid world
%
%Rew_str, Nx1, specifies the rewards of different states
%transition, NxN, transition matrix
%par includes all the parameters
%plotting should be 'nope'(default), 'step', 'pstep' or 'epis'
%
%Note that if gridworld is of size m x n, then transitions is of size (m*n) x (m*n)
%Choices are made through e-greedy noise model
figure(1);close(1)

if nargin<4
    plotting='nope';
end

nepisodes=1000;

%number of states
N=length(Rew_str);

%initialize q-value for all NxN state-transition pairs
Q=transition+rand(size(transition))/1000; %The initial action values are given by the possible transitions
cumr(1:nepisodes)=0;
i=0;
while  i<nepisodes%run up to nepisodes
    i=i+1;
    s=4;%start position
    [temp,a]=max(Q(s,:));
    if strcmp(plotting,'step')%plot at every step
        gridworld_plotN(Q,Rew_str,s)
    end
    if strcmp(plotting,'pstep')%plot at every step, then pause
        gridworld_plotN(Q,Rew_str,s)
        pause
    end
    
    while Rew_str(s)~=0 & Rew_str(s)~=-100%i.e. stop position not reached
        [temp,a]=max(Q(s,:));
        chance=par(1)>rand(1); %random binary, (1-i/1000)*par(1)<rand(1)
        options=find(Q(s,:)>-Inf);%only allow choices with finite value

        a=(1-chance)*a + chance*options(ceil(rand(1)*length(options)));%par(1) chance of picking randomly

        
        %note that in this gridworld the action is moving to the new location s'
        %Q(s,a) is therefore a transition matrix from s to a=s'
        sn=a;
        
        %INSERT ALGORITHM
        %update Q with new action a
        Q(s,a)=Q(s,a)+par(3)*(Rew_str(sn)+par(2)*max(Q(sn,:))-Q(s,a));
%         a_options=find(Q(sn,:)>-Inf);
%         an=a_options(randi(length(a_options)));
%         Q(s,a)=Q(s,a)+par(3)*(Rew_str(sn)+par(2)*Q(sn,an)-Q(s,a));
        
        cumr(i)=cumr(i)+Rew_str(sn);
        
        s=sn;%old state s becomes new state sn
        
        
        if strcmp(plotting,'step')%plot at every step
            gridworld_plotN(Q,Rew_str,s)
        end
        if strcmp(plotting,'pstep')%plot at every step, then pause
            gridworld_plotN(Q,Rew_str,s)
            pause
        end
    end
    Qhist(:,:,i)=Q;
    disp(sprintf('episode  %d   rewards  %d  ', [i cumr(i)]))
    if strcmp(plotting,'epis')%plot at every episode
        gridworld_plotN(Q,Rew_str)
    end
    
end

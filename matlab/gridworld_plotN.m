function gridworld_plot(V,Rew_str,position)
%Written by Ulrik Beierholm
N=sqrt(length(V));
figure(1)
hold off
colormap gray
if size(V,2)==1
ims=(reshape(V,N,N));
else
ims=(reshape(max(V')',N,N));
end  
temp=max(ims');
while temp(end)==-Inf;
ims=ims(1:end-1,:);
temp=max(ims');
end

imagesc(ims)
%image((ims+2)*60)
if nargin>2
    hold on
    plot(ceil(position/N), mod(position-1,N)+1,'*b','MarkerSize',30,'Linewidth',10)
end
if nargin>1%indicate good and bad
    temp=find(Rew_str==0);
    hold on
    plot(ceil(temp/N), mod(temp-1,N)+1,'+g','Markersize',30,'Linewidth',10)
    
    temp=find(Rew_str==-100);
    hold on
    plot(ceil(temp/N), mod(temp-1,N)+1,'or','MarkerSize',30,'Linewidth',10)
    
end
colorbar
pause(0.1)

function plaintable_plot(object_pos,particle_pos,[0 20 0 20])
    figure(1);
    hold off;
    hold on;
    plot(object_pos(1),object_pos(2),'red*','Markersize',10,'Linewidth',3)


    hold on;
    plot(particle_pos(:,1),particle_pos(:,2),'blue.');
    axis([0 20 0 20] )




end
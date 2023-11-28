function DoMyViz_node_Epilepsy(cortex, idx_atlas, Mean_Node_Diff_group, filename, vmin, vmax)

    close all;
    N = length(cortex.Atlas(idx_atlas).Scouts);
    col = zeros(N,3);

   figure(1)
   subplot(2,2,1)
   p=Mean_Node_Diff_group; 
   val_max=vmax;
   val_min=vmin;
    
   for i=1:N
        p_scaledValue(i) = (p(i)-val_min) /(val_max-val_min);
   end


    for i=1:N
         nver = length(cortex.Atlas(idx_atlas).Scouts(i).Vertices);
         col(cortex.Atlas(idx_atlas).Scouts(i).Vertices,:) = repmat(cmap(p_scaledValue(i),'summer'),nver,1);
    end
    colmap = cmap(64,'summer');
    set(gcf,'color','w')
    hp2 = patch('faces',cortex.Faces,'vertices',cortex.Vertices,'FaceVertexCData',col);
    set(hp2,'linestyle','none','FaceColor', 'interp','specularstrength',0);
    axis equal tight off
    alpha(1)
    light('position',[-1 0 0]);
    light('position',[1 1 0]);
    light('position',[0 0 1]);
    lighting gouraud
    view([-180 0]); % left
    colormap(colmap);
    un=unique(p);
    subplot(2,2,2)
   p=Mean_Node_Diff_group; 
   val_max=vmax;
   val_min=vmin;
    
   for i=1:N
        p_scaledValue(i) = (p(i)-val_min) /(val_max-val_min);
   end


    for i=1:N
         nver = length(cortex.Atlas(idx_atlas).Scouts(i).Vertices);
         col(cortex.Atlas(idx_atlas).Scouts(i).Vertices,:) = repmat(cmap(p_scaledValue(i),'summer'),nver,1);
    end
    colmap = cmap(64,'summer');
    set(gcf,'color','w')
    hp2 = patch('faces',cortex.Faces,'vertices',cortex.Vertices,'FaceVertexCData',col);
    set(hp2,'linestyle','none','FaceColor', 'interp','specularstrength',0);
    axis equal tight off
    alpha(1)
    light('position',[1 0 0]);
    light('position',[-1 -1 0]);
    light('position',[0 0 1]);
    lighting gouraud
    view([-90 90]); % up
    view([0 360]); % right
    colormap(colmap);
    un=unique(p);
    subplot(2,2,3)
       p=Mean_Node_Diff_group; 
   val_max=vmax;
   val_min=vmin;
    
   for i=1:N
        p_scaledValue(i) = (p(i)-val_min) /(val_max-val_min);
   end


    for i=1:N
         nver = length(cortex.Atlas(idx_atlas).Scouts(i).Vertices);
         col(cortex.Atlas(idx_atlas).Scouts(i).Vertices,:) = repmat(cmap(p_scaledValue(i),'summer'),nver,1);
    end
    colmap = cmap(64,'summer');
    set(gcf,'color','w')
    temp=cortex.Vertices;
    test=[];test2=[];
    test=temp(:,2);
    test2=find(test>0.00045);
    test(test2)=NaN;
    temp2=[cortex.Vertices(:,1),test,cortex.Vertices(:,3)];

    hp2=patch('faces',cortex.Faces,'vertices',temp2,'FaceVertexCData',col);
    set(hp2,'linestyle','none','FaceColor', 'interp','specularstrength',0);
    axis equal tight off
    alpha(1)
    light('position',[-1 0 0]);
    light('position',[1 1 0]);
    light('position',[0 0 1]);
    lighting gouraud
    view([-180 0]); % left
    colormap(colmap);
    un=unique(p);
  
   subplot(2,2,4)
   p=Mean_Node_Diff_group; 
   val_max=vmax;
   val_min=vmin;
    
   for i=1:N
        p_scaledValue(i) = (p(i)-val_min) /(val_max-val_min);
   end


    for i=1:N
         nver = length(cortex.Atlas(idx_atlas).Scouts(i).Vertices);
         col(cortex.Atlas(idx_atlas).Scouts(i).Vertices,:) = repmat(cmap(p_scaledValue(i),'summer'),nver,1);
    end
    colmap = cmap(64,'summer');
    set(gcf,'color','w')
            temp=cortex.Vertices;
        test=[];test2=[];
        test=temp(:,2);
        test2=find(test<-0.00135);
        test(test2)=NaN;
        temp2=[cortex.Vertices(:,1),test,cortex.Vertices(:,3)];
        
    hp2=patch('faces',cortex.Faces,'vertices',temp2,'FaceVertexCData',col);
    set(hp2,'linestyle','none','FaceColor', 'interp','specularstrength',0);
    axis equal tight off
    alpha(1)
    light('position',[1 0 0]);
    light('position',[-1 -1 0]);
    light('position',[0 0 1]);
    lighting gouraud
    view([0 360]); % right
    colormap(colmap);
    un=unique(p);
    saveas(gcf,strcat(filename,'_test.pdf'));
    savefig(gcf,strcat(filename,'_test.fig'));
    saveas(gcf,strcat(filename,'.epsc'));
end
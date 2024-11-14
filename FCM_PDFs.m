
clear all
clc
 x=0.01:.01:13;
% Define the probability density functions
% p = @(x) normpdf(x, 0.2, 2);  % Normal distribution with mean 0 and standard deviation 1
% q = @(x) normpdf(x, 1.1, 1);  % Normal distribution with mean 1 and standard deviation 1
%f1 = @(x) normpdf(x,0.2,1);
load data12a
[center, U, obj_fcn, dist] = fcmPDF(dataa, 4)

for i=1:12
     da(i,:)=plot(x,dataa(i,:),'--r', 'LineWidth',1);
     set(gcf,'color','w');
     hold on
end
for j=1:3
hold on
     mn(j,:)=plot(x,center(j,:),'-b','LineWidth',1);
     set(gcf,'color','w');
     xlabel('x')
ylabel('g(x)')
legend([da(1,:) mn(1,:)],{'input PDFs',' Updating PDFs'});
end
 hold off




FU=(1/size(U,2))*sum(sum(U,2))
DFC= (FU-(1/4))/(1-(1/4))

% Example fuzzy matrix (3 rows and 12 columns)
fuzzy_matrix = U';  % Example random fuzzy matrix

% Create meshgrid for x and y coordinates
[x, y] = meshgrid(1:size(fuzzy_matrix, 2), 1:size(fuzzy_matrix, 1));

% Plotting the fuzzy matrix as a 3D bar plot
figure;
bar3(fuzzy_matrix);
colormap('jet');  % Using the 'jet' colormap
colorbar;
xlabel('Cluster');
ylabel('PDFs');
zlabel('Membership degree');
title('Fuzzy Matrix');

% Add text annotations with the values
for i = 1:numel(fuzzy_matrix)
    text(x(i), y(i), fuzzy_matrix(i), num2str(fuzzy_matrix(i), '%.1f'), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

close all;
clear all;
clc;

classes=30;
%images=13*30;%70 for training, 30 for testing
%images=300; %700 for training, 300 for testing
images=7; %18 for training, 7 for testing
fid=fopen('sketch_test_pair.txt','w');

for class=1:1:classes
    for image=1:1:images
        imagename=strcat(num2str(image+(class-1)*images),'.png');
        fprintf(fid,'%s %d\r\n',imagename, class-1);
    end
    
end


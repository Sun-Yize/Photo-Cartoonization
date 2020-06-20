clc,clear,close all;

img = imread('image path here');
figure,subplot(2,2,1),imshow(img);
title('原始图像')

img = double(img)/255;
result = bilateral_filter(double(img),8,[3 0.1]);
subplot(2,2,2),imshow(result);
title('双边滤波后')

result = edge_detect(result);
subplot(2,2,3),imshow(result);
title('边缘加重')

result = color_adjust(result);
subplot(2,2,4),imshow(result);
title('颜色调整')
imwrite(result,'result.bmp')
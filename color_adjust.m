% 图像饱和度调整
function img_res = color_adjust(img)
    img_double = double(img);
    img_gray = double(rgb2gray(img));
    S_template = img_double;
    % 使用原始图像的灰度图像作为模板，来完成插值和外推
    S_template(:,:,1)=img_gray; 
    S_template(:,:,2)=img_gray;
    S_template(:,:,3)=img_gray;
    % 插值和外推 
    img_res = (-1).*S_template  +2.*img_double;
    img_res = uint8(img_res);
end
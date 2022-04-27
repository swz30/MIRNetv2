
close all;clear all;

scales = {2,3,4};
num_scales = length(scales);
for idx_set = 1:num_scales
    file_path = strcat('./results/realSR/x', num2str(scales{idx_set}), '/');
    gt_path = strcat('./Datasets/test/realSR/x', num2str(scales{idx_set}), '/target/');
    path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
    gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
    img_num = length(path_list);
    total_psnr = 0;
    total_ssim = 0;
    scale = scales{idx_set};
    if img_num > 0 
        for j = 1:img_num 
           image_name = path_list(j).name;
           gt_name = gt_list(j).name;
           input = imread(strcat(file_path,image_name));
           gt = imread(strcat(gt_path, gt_name));

           input = input(scale:end-scale,scale:end-scale,:);
           gt = gt(scale:end-scale,scale:end-scale,:);
           
           input_ycbcr = rgb2ycbcr(input);
           input_y = input_ycbcr(:, :, 1);
           
           gt_ycbcr = rgb2ycbcr(gt);
           gt_y = gt_ycbcr(:, :, 1);
           
           ssim_val = ssim_index(input_y, gt_y);
           psnr_val = psnr(input_y, gt_y);
           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
       end
    end
    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;
    fprintf('For scale factor %d PSNR: %f SSIM: %f\n', scales{idx_set}, qm_psnr, qm_ssim);
    
end


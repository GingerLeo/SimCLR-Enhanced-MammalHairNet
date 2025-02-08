rootFolder = '/home/lzh/hair/hair_2025/hair_2025_per15';

% 获取根文件夹下的所有子文件夹
subFolders = dir(rootFolder);
subFolders = subFolders([subFolders.isdir]); % 仅保留文件夹项

% 遍历每个子文件夹
for i = 1:length(subFolders)
    folderName = subFolders(i).name;
    
    % 忽略.和..文件夹
    if strcmp(folderName, '.') || strcmp(folderName, '..')
        continue;
    end
    
    % 构建子文件夹的完整路径
    subFolderPath = fullfile(rootFolder, folderName);
    
    % 获取子文件夹中的所有.TIF文件
    tifFiles = dir(fullfile(subFolderPath, '*.Tif'));
    
    % 遍历每个.TIF文件并进行裁剪操作
    for j = 1:length(tifFiles)
        tifFileName = tifFiles(j).name;
        
        % 构建.TIF文件的完整路径
        tifFilePath = fullfile(subFolderPath, tifFileName);
        
        % 读取.TIF文件
        image = imread(tifFilePath);
        
        % 获取图像的原始高度和宽度

        [original_height, original_width, ~] = size(image);
        
        % 计算要裁剪的高度
        crop_height = round(original_height * 0.062);
        
        % 从底部向上裁剪图像
        cropped_image = image(1:end-crop_height, :, :);
        
        % 计算裁剪后图像的新高度和宽度
        [new_height, new_width, ~] = size(cropped_image);
        
        % 计算居中裁剪的起始位置
        start_x = floor((new_width - new_height) / 2) + 1;
        end_x = start_x + new_height - 1;
        
        % 居中裁剪成一个正方形
        center_cropped_image = cropped_image(:, start_x:end_x, :);


        % 保存裁剪后的图像
        imwrite(center_cropped_image, tifFilePath);
    end
end
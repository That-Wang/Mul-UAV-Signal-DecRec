% Constants
global pi fs fc0 showfig;
pi = 3.141592653589793;
fs = 100000000;
fc0 = 8000000;
showfig = 0;

% 文件夹路径
labelsDir = 'D:\WPS Yunpan\WPS云盘\01研究生\神经网络\RFnet\uav_rec_lsnr_ICallDA2\labels_fixsnr_dec'; % 'D:\WPS Yunpan\WPS云盘\01研究生\神经网络\RFnet\uav_rec_lsnr_ICallDA2\labels_dec';
matDir = 'D:\download\UAVDataset\alltest_fixsnr';  % 'D:\download\UAVDataset\alltest'; 
img_directDir = 'D:\download\UAVDataset\alltest\directimg';
img_Dir = 'D:\download\UAVDataset\alltest_fixsnr\images';
img_yoloDir = 'D:\download\UAVDataset\alltest_fixsnr\yoloimg';

% 获取所有标签文件
labelFiles = dir(fullfile(labelsDir, '*.txt'));

% 遍历每个标签文件
for k = 1:length(labelFiles)
    labelFile = labelFiles(k).name;
    labelPath = fullfile(labelsDir, labelFile);
    
    % 获取对应的 .mat 文件名
    baseName = strrep(labelFile, 'UAVSig', 'UAVSig');  % strrep(labelFile, 'Image', 'IQ');
    baseName = strrep(baseName, '.txt', '.mat');
    matPath = fullfile(matDir, baseName);

    OneImageName0 = strrep(labelFile, 'UAVSig', 'OneUAVImage');  % strrep(labelFile, 'Image', 'OneImage');
    OneImageName0 = strrep(OneImageName0, '.txt', '.png');  % strrep(OneImageName0, '.txt', '.png');
    OneIQName0 = strrep(labelFile, 'UAVSig', 'OneUAVIQ');  % strrep(labelFile, 'Image', 'OneIQ');
    OneIQName0 = strrep(OneIQName0, '.txt', '.mat');  % strrep(OneIQName0, '.txt', '.mat');
    FeatName0 = strrep(labelFile, 'UAVSig', 'OneUAVFeat');  % strrep(labelFile, 'Image', 'Feat');
    FeatName0 = strrep(FeatName0, '.txt', '.mat');  % strrep(FeatName0, '.txt', '.mat');
    
    % 检查对应的 .mat 文件是否存在
    if ~isfile(matPath)
        fprintf('未找到 %s 的 MAT 文件\n', labelFile);
        continue;
    end
    
    % 从 .txt 文件中读取位置信息
    fileID = fopen(labelPath, 'r');
    lines = textscan(fileID, '%s', 'Delimiter', '\n');
    fclose(fileID);
    lines = lines{1};
    
    % 加载 .mat 文件
    matData = load(matPath);
    data1 = matData.FinalSig;  % data1
    [fa, ta, Pa] = abstft(data1, fs, 2048, 0);
    PPa = minmax(Pa) * 255;
    ima = mat2gray(PPa);
    len = length(data1);
    % 处理 .txt 文件中的每个目标
    for i = 1:length(lines)
        % 保存的文件名
        OneImageName = strrep(OneImageName0, '.png', sprintf('-%d.png', i));
        OneIQName = strrep(OneIQName0, '.mat', sprintf('-%d.mat', i));
        FeatName = strrep(FeatName0, '.mat', sprintf('-%d.mat', i));
        % 提取位置信息（假设格式：class x_center y_center width height）
        label = sscanf(lines{i}, '%f');
        class_id = label(1);
        x_center = label(2);
        y_center = label(3);
        width = label(4);
        height = label(5);
        startpoint = round((len*x_center - len*width/2)+1);
        endpoint = round((len*x_center + len*width/2)-1);
        fc = fs*y_center - fs/2;
        %fc = round(fc/1e6)*1e6;
        bw = fs*height; %10e6; % 

        % 下变频滤波+裁剪
        ccos = cos(2 * pi * -fc / fs * (0:len-1));
        csin = sin(2 * pi * -fc / fs * (0:len-1));
        temp = data1 .* (ccos + 1i * csin);
%         [f, t, P] = abstft(temp, fs, 2048, 512);
        rsig = bl_filter(temp, bw/2, fs, 8);
%         [f, t, P] = abstft(rsig, fs, 2048, 512);
        SepSig = rsig(startpoint:endpoint);
        % 保存单信号时频图
        [f, t, P] = abstft(SepSig, fs, 2048, 0);
        PP = minmax(P) * 255;
        im = mat2gray(PP);
        heightim = size(im, 1);
        widthim = size(im, 2);
        if heightim > 1536 % 确保高度大于 1024，以便裁剪上下各 768 像素
            sub = round(2048*(1-bw/fs)/2); % sub = 768; % 
            im_cropped = im(sub+1:heightim-sub, :); % 裁剪上下各 768 像素
        else
            error('图像高度不足以裁剪。');
        end
        imwrite(im_cropped, fullfile(img_Dir, OneImageName));

%% 保存子图像
        [currentHeight, currentWidth, ~] = size(im_cropped);
        targetHeight = 512;
        targetWidth = 512;
        padTop = floor((targetHeight - currentHeight) / 2);
        padBottom = ceil((targetHeight - currentHeight) / 2);
        padLeft = floor((targetWidth - currentWidth) / 2);
        padRight = ceil((targetWidth - currentWidth) / 2);
        im_yolocropped = padarray(im_cropped, [padTop, padLeft], 0, 'pre');
        im_yolocropped = padarray(im_yolocropped, [padBottom, padRight], 0, 'post');
        imwrite(im_yolocropped, fullfile(img_yoloDir, OneImageName));

%% 直接从yolo原始图像上剪切信号
%         [height_im, width_im, ~] = size(ima);
% %         print('height_im:%d, width_im:%d',height_im, width_im)
% 
%         % 将归一化坐标转换为像素坐标
%         x_center = round(x_center * width_im);
%         y_center = round(y_center * height_im);
%         width_pixels = round(width * width_im);
%         height_pixels = round(height * height_im);
%         % 计算边界框的左上角和右下角坐标
%         x_min = max(1, x_center - floor(width_pixels / 2));
%         y_min = max(1, y_center - floor(height_pixels / 2));
%         x_max = min(width_im, x_min + width_pixels - 1);
%         y_max = min(height_im, y_min + height_pixels - 1);
%         % 裁剪图像
%         cropped_object = ima(y_min:y_max, x_min:x_max, :);
%         [height_cropped_object, width_cropped_object, ~] = size(cropped_object);
%         fprintf('height_cropped_object: %d,\t width_cropped_object: %d\n',height_cropped_object, width_cropped_object)
%         imwrite(cropped_object, fullfile(img_directDir, OneImageName));
% %         % 显示裁剪的目标（可选）
% %         figure;
% %         imshow(cropped_object);
% %         title(['cropped_Object ', num2str(k)]);

%% 其它特征        
%         % 时域特征
%         feat = features(SepSig); % Feature extraction
%         % 保存单信号时域特征
%         save(fullfile(matDir, FeatName), 'feat');
%         resig = single(resig);                   
%         % 保存单信号IQ序列
%         save(fullfile(matDir, OneIQName), 'SepSig');

    end
    fprintf('处理了 %s，包含 %d 个目标。\n', labelFile, length(lines));
end
close("all");
clear;


% Butterworth lowpass filter
function y = bl_filter(data, cutoff, fs, order)
    [b, a] = butter(order, cutoff / (0.5 * fs), 'low');
    y = filter(b, a, data);
end

% Short-Time Fourier Transform (STFT)
function [f, t, P] = abstft(data1, fs, nperseg, overlap)
    [S, f, t] = stft(data1, fs, Window=hamming(nperseg), OverlapLength = overlap, FFTLength = nperseg);
    P = abs(S);
    global showfig
    if showfig == 1
        figure;
        % surf(t, f, 10*log10(P), 'EdgeColor', 'none');
        imagesc(t, f, P);
        axis xy; axis tight; %colormap(jet); %view(0, 90);
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title('Spectrogram');
    end
end

function feat = features(x1)
    % Split the input signal into two halves
%     x1 = x(1:round(length(x)/2));
%     x2 = x(round(length(x)/2)+1:end);    
    % Initialize the feature vector
    feat = zeros(1, 8);    
    % Calculate features for the first half
    feat(1) = skewness(x1); % Skewness
    feat(2) = kurtosis(x1); % Kurtosis
    feat(3) = var(x1); % Variance
    feat(4) = max(abs(x1)) - min(abs(x1)); % Peak-to-peak
    feat(5) = rms(x1) / mean(abs(x1)); % Shape factor
    feat(6) = max(abs(x1)) / rms(x1); % Crest factor
    feat(7) = max(abs(x1)) / mean(abs(x1)); % Impulse factor
    feat(8) = max(abs(x1)) / (mean(sqrt(abs(x1)))^2); % Clearance factor 
    % Calculate features for the second half
%     feat(9) = skewness(x2); % Skewness
%     feat(10) = kurtosis(x2); % Kurtosis
%     feat(11) = var(x2); % Variance
%     feat(12) = max(abs(x2)) - min(abs(x2)); % Peak-to-peak
%     feat(13) = rms(x2) / mean(abs(x2)); % Shape factor
%     feat(14) = max(abs(x2)) / rms(x2); % Crest factor
%     feat(15) = max(abs(x2)) / mean(abs(x2)); % Impulse factor
%     feat(16) = max(abs(x2)) / (mean(sqrt(abs(x2)))^2); % Clearance factor
end

function y = minmax(x)
    maxVal = max(x(:));
    minVal = min(x(:));
    y = (x - minVal) / (maxVal - minVal);
end



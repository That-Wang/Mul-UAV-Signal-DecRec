% % 倒序执行修改文件名
% % 指定文件夹路径
% folderPath = 'D:\download\UAVDataset\inspire2Vid'; % 替换为你的文件夹路径
% % 获取文件夹中所有符合条件的文件
% files = dir(fullfile(folderPath, 'inspire2s_*.mat'));
% for i = length(files):-1:1
%     % 提取原文件名和路径
%     oldName = files(i).name;
%     oldFullPath = fullfile(folderPath, oldName);
%     
%     % 提取编号部分
%     numStr = regexp(oldName, '\d+', 'match');
%     num = str2double(numStr{end}); % 使用最后一个匹配的数字
%     
%     % 生成新的文件名和路径
%     newName = sprintf('inspire2_%d.mat', num);
%     newFullPath = fullfile(folderPath, newName);
%     
%     % 重命名文件
%     movefile(oldFullPath, newFullPath);
% end


%(-500~500)   inspire2:-385~-289(200ms,9.6MHz,-335e5Hz)  mini2:-358~-176(200ms,18.2MHz,-265e5Hz)
%matrice:14.5~196.5(100ms,18.2MHz,105.5e5Hz)  mavic:-378~-196(1000ms,18.2MHz,-287e5Hz)
global fs showfig pi
pi = 3.141592653589793;
fs = 100e6;
showfig = 1;
% 指定文件夹路径
folderPath = 'D:\\download\\UAVDataset\\mavicVid1'; % 替换为你的文件夹路径
uavname =  'mavic';
% 获取文件夹中所有符合条件的文件
files = dir(fullfile(folderPath, sprintf('%s_*.mat', uavname)));
fcc = -287e5;
bw = 20e6;
count = 0;  % 计数
% 遍历每个文件
for i = 1:12 %length(files)
    % 构建完整文件路径
    filePath = fullfile(folderPath, files(i).name);
    % 加载文件
    data = load(filePath);   
    % 假设数据在变量 'dataVar' 中，替换为实际的变量名
    % 如果文件中有多个变量，可以使用 fieldnames(data) 来查看变量名称
    varName = fieldnames(data);
    OriginSig = data.(varName{1});

    fc = -fcc; %pmidfre(BaseSig, fs, fcc);

    % 下变频滤波
%     [f, t, P]=abstft(BaseSig.', fs, 2048, 0);
%     title('下变频前信号')

    c = exp(1i*2*pi*fc*(1:length(OriginSig))/fs);
    OriginSig = OriginSig.*c.';  
%     [f, t, P]=abstft(BaseSig.', fs, 2048, 0);
%     title('下变频后信号')
%     PP = minmax(P) * 255;
%     im = mat2gray(PP);
%     figure;
%     imshow(im);
%     title('下变频滤波后信号')
    OriginSig = bl_filter(OriginSig, bw/2, fs, 10);
    [f, t, P]=abstft(OriginSig.', fs, 2048, 0);
    title('下变频滤波后信号')
    
    % 双滑动窗口信号检测
    windowSize = 100; % 窗口大小
    ratioThreshold = 0.1; % 比率阈值 inspire2:0.2, mini2:0.04, matrice:0.04, mavic:0.1
    skipSize = 2*windowSize;
    % 初始化
    signalLength = length(OriginSig);
    startPoints = [];
    endPoints = [];
    % 滑动窗口检测
    j = 1;
    while j <= (signalLength - 2 * windowSize + 1)
        A = sum(abs(OriginSig(j:j+windowSize-1)));
        B = sum(abs(OriginSig(j+windowSize:j+2*windowSize-1)));       
        % 检测起始点
        if A / B < ratioThreshold
            startPoints = [startPoints, j + windowSize];
            j = j + skipSize; % 跳过检测到的区域
            continue;
        end        
        % 检测终止点
        if B / A < ratioThreshold
            endPoints = [endPoints, j + windowSize];
            j = j + skipSize; % 跳过检测到的区域
            continue;
        end        
        j = j + 10; % 正常滑动
    end  
    % 绘制波形图
    figure;
    plot(real(OriginSig), 'b');
    hold on;    
    % 标记起始和终止点
    for k = 1:length(startPoints)
        % 标记起始点
        plot(startPoints(k), real(OriginSig(startPoints(k))), 'go', 'MarkerFaceColor', 'g');
    end    
    for k = 1:length(endPoints)
        % 标记终止点
        plot(endPoints(k), real(OriginSig(endPoints(k))), 'ro', 'MarkerFaceColor', 'r');
    end    
    xlabel('Sample Index');
    ylabel('Amplitude');
    title('Signal with Start and End Points');
    legend('Signal', 'Start Points', 'End Points');
    hold off; 

    % 保存分割后的信号段并打印起止信息
    fid = fopen(sprintf('%s\\%s_%d.txt', folderPath, uavname, i), "a");
    segments = {};
    m = 1; n = 1;    
    while m <= length(startPoints) && n <= length(endPoints)
        if startPoints(m) < endPoints(n)
            % 有效的信号段
            segments{end+1} = OriginSig(startPoints(m):endPoints(n));
            fprintf(fid, '%d %d %d %d\n', m, startPoints(m), endPoints(n), bw); 
            m = m + 1;
            n = n + 1;
        else
            % 如果终止点在当前起始点之前，跳过终止点
            n = n + 1;
        end
    end
    fclose(fid);

    for k = 1:length(segments)
        count = count+1;
        TempSig = segments{k};
        startIndex = find(abs(TempSig) > 0.01, 1, 'first');
        BaseSig = TempSig(startIndex:end);
        figure;
        plot(real(BaseSig));
        title(sprintf('%s_%d的第%d个分割信号', uavname, i, k))
%         abstft(BaseSig.', fs, 2048, 0);
        filename = fullfile(folderPath, sprintf('%s-%d.mat', uavname, count));
        save(filename, 'BaseSig');
        fprintf('Signal %d, Segment %d, length: %.2fms\n', i, k, length(BaseSig)/(fs/1000));
    end
end
close("all");


% Butterworth lowpass filter
function y = bl_filter(data, cutoff, fs, order)
    [b, a] = butter(order, cutoff / (0.5 * fs), 'low');
    y = filter(b, a, data);
end

function [f, t, P] = abstft(data1, fs, nperseg, overlap)
    [S, f, t] = stft(data1, fs, Window=hamming(nperseg), OverlapLength = overlap, FFTLength = nperseg);
    P = abs(S);
    global showfig
    if showfig == 1
        figure;
        % surf(t, f, 10*log10(P), 'EdgeColor', 'none');
        imagesc(1:length(t), f, P);
        axis xy; axis tight; %colormap(jet); %view(0, 90);
        xlabel('时间点');
        ylabel('频率点');
        title('时频图');
    end
end

    % 基于功率谱滑动窗口计算中心频点
function midfre = pmidfre(BaseSig, fs, fcc)
    [pxx, f] = pwelch(BaseSig, 2048, 0, 2048, fs);
    maxSum = 0;
    centerFreq = 0;
    for startIdx = 1:10:(length(pxx) - 100 + 1)
        segxx = pxx(startIdx:startIdx + 100 - 1);
        currentSum = sum(segxx);
        if currentSum > maxSum
            maxSum = currentSum;
            % 找到当前段的最大值对应的频率
            [~, localIdx] = max(segxx);
            centerFreq = f(startIdx + localIdx - 1);
        end
    end
    centerFreq = centerFreq - fs;
    if abs(centerFreq-fcc) < 2*fs/2048
        midfre = -centerFreq;
    else
        midfre = -fcc;
    end
    % 打印中心频点
    fprintf('中心频点 (滑动窗口PSD): %.2f Hz, 使用中: %.2f Hz\n', centerFreq, fc);
end

function y = minmax(x)
    maxVal = max(x(:));
    minVal = min(x(:));
    y = (x - minVal) / (maxVal - minVal);
end

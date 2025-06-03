% Constants
global pi fs fc0 showfig;
pi = 3.141592653589793;
fs = 100e6;
fc0 = 10e6;
showfig = 0;
totalSig = 140;  % 信号总数 84 28 28 ->42 14 14
numPerSig = 5;  % 接收信号包含5个个体
snrpool = [-15,-10,-5,0,5,10];  % 信噪比
snrfix = [-5, 0, 5, 10]; % 固定信噪比,[-5, 0, 5, 10]
len = fs/100; % 信号长度:1/100(s)
ofdmLens10m = 9.76e5;
ofdmLens20m = [1e5, 1.07e5, 2e5, 2.07e5, 3e5, 3.11e5, 4e5, 4.07e5, 5e5, 5.11e5];
offset = true;  % 信号扰动

DatasetNames = {'DJI M100(1)','DJI M100(2)','DJI M100(3)','DJI M100(4)','DJI M100(5)', 'OFDM', 'DJI Inspire2', 'DJI mini2', 'DJI Matrice pro', 'DJI Mavic'}; %{'DJI M100(1)','DJI M100(2)','DJI M100(3)','DJI M100(4)','DJI M100(5)', 'OFDM'};%
DatasetInfos = selectRandomDatasets(DatasetNames,numPerSig,totalSig,false);  % moreOFDM = false
numSample = length(DatasetInfos);

% Main processing loop
for i = round(numSample/10*8)+1:numSample % 一个信号集合的时频图
    if i <= round(numSample/10*6)
        DA_freshift = true;  % 信号扰动数据增强
        DA_snr = true;  % 信号信噪比组合数据增强
    else
        DA_freshift = false;
        DA_snr = true;
    end
%% 初步生成数据集,准备原始样本集合
    SigSet0 = cell(numPerSig,1);
    SigSet = cell(numPerSig,1);
    fclist = zeros(numPerSig,1);
    bwlist = zeros(numPerSig,1);
    snrlist = zeros(numPerSig,1);
    labellist = zeros(numPerSig,1);
    sigLen = zeros(numPerSig,1);

    for j = 1:numPerSig % 一个信号集合时频图中的单个信号索引
        switch DatasetInfos(i).labels{j}
            case {'0','1','2','3','4'}
                labellist(j) = str2double(DatasetInfos(i).labels{j});
                bwlist(j) = DatasetInfos(i).bws(j);
                orisig = origininput(DatasetInfos(i).filenames{j})/100; % Import original signal
                BaseSig = upsampleSig(orisig); % Upsample
                BaseSig = bl_filter(BaseSig, bwlist(j)/2, fs, 8); % Filter
                [SigSet0{j}, fclist(j)] = rf_modu(BaseSig,bwlist(j),j); % Frequency shift
                sigLen(j) = length(SigSet0{j});
            case {'6','7','8','9'}
                labellist(j) = str2double(DatasetInfos(i).labels{j});
                bwlist(j) = DatasetInfos(i).bws(j);
                load(DatasetInfos(i).filenames{j});
                [SigSet0{j}, fclist(j)] = rf_modu(BaseSig.',bwlist(j),j); % Frequency shift
                sigLen(j) = length(SigSet0{j});
            case '5'
                labellist(j) = str2double(DatasetInfos(i).labels{j});
                bwlist(j) = DatasetInfos(i).bws(j);
                % 找到 sigLen 中大于 0 的元素
                indicesLen = find(sigLen > 0);
                if ~isempty(indicesLen)
                    % 从大于 0 的元素中随机选择一个
                    indexLen = datasample(indicesLen, 1);
                    alen = sigLen(indexLen);
                    bwlist(j) = bwlist(indexLen);
%                     if bwlist(j) == 20e6
%                         bwtemp = 18.2e6;
%                     elseif bwlist(j) == 10e6
%                         bwtemp = 9.6e6;
%                     end
                else
                    % 如果没有大于 0 的元素，从 ofdmBws 中随机选择一个
                    if bwlist(j) == 18.2e6
                        alen = ofdmLens20m(randi(length(ofdmLens20m)));
                    elseif bwlist(j) == 9.6e6
                        alen = ofdmLens10m;
                    end
                end
                BaseSig = (GenOfdm(fs,bwlist(j),alen)).'; % 生成Ofdm成型滤波信号 alen为信号长度
                [SigSet0{j}, fclist(j)] = rf_modu(BaseSig,bwlist(j),j); % Frequency shift
        end
    end

    % 裁剪所有信号不大于10ms
    for j = 1:numPerSig
        if length(SigSet0{j}) > len  % 10ms
            SigSet0{j} = double(SigSet0{j}(1:len));
        else
            SigSet0{j} = double(SigSet0{j});
        end
    end
%     for j = 1:numPerSig
%         abstft(SigSetArray(j,:),fs,2048, 512);
%     end
%% 生成数据集与标签
    if DA_snr == true
        da_snr_num = length(snrpool);
    else
        da_snr_num = 1;
    end
    for ak = 1:numPerSig % 一个信号集合的复用次数, ak即为具有动态信噪比的信号的索引
        for m = 1:da_snr_num
            x_center = zeros(numPerSig, 1);
            y_center = zeros(numPerSig, 1);
            width = zeros(numPerSig, 1);
            height = zeros(numPerSig, 1);
            % mask = zeros(size(P));
            startpoint = zeros(numPerSig, 1);
            endpoint = zeros(numPerSig, 1);
            for k = 1:numPerSig  % 变换信号信噪比，保存信号定位信息
                if k == ak % 跟踪其中一个信号的信噪比变化
                    if m == 1  % 遍历所有snr
                        snrlist(ak) = randsample(snrpool,1);
                    else
                        current_snr = snrlist(k);
                        index_snr = find(snrpool == current_snr, 1);
                        if index_snr < length(snrpool)
                            snrlist(k) = snrpool(index_snr + 1);  % Move to the next value
                        else
                            snrlist(k) = snrpool(k);
                        end
                    end
                elseif k < ak
                    snrlist(k) = snrfix(k);
                else
                    snrlist(k) = snrfix(k-1);
                end
                SigSet{k} = ChangePower(SigSet0{k}, snrlist(k)); % 调整信号功率
    
                % 计算需要补零的数量
                len0 = length(SigSet{k});
                numZeros = len - len0;            
                if numZeros > 0
                    % 补零扩展信号
                    SigSet{k} = [SigSet{k} zeros(1, numZeros)];
                end
                shift = randi([0, numZeros]); 
                SigSet{k}(shift+1:shift+len0) = SigSet{k}(1:len0);
                SigSet{k}(1:shift) = 0; 
        
                startpoint(k) = shift + 1;
                endpoint(k) = shift + len0;
                x_center(k) = (shift + (len0+1)/2) / length(SigSet{k}); % 定位信号
                y_center(k) = (fclist(k) + fs/2) / fs;
                width(k) = len0 / length(SigSet{k});
                height(k) = bwlist(k) / fs;
                if k == 1
                    writeMark = 'w';
                else
                    writeMark = 'a';
                end
    %             % 保存yolo标签!!!!!!
                if i <= round(numSample/10*6)
                    continue
    %                 fid_dec = fopen(sprintf('D:\\download\\UAVDataset\\dectrain\\labels_dec\\UAVSig%d_DA%d.txt', i, m), writeMark);
    %                 fid_decrec = fopen(sprintf('D:\\download\\UAVDataset\\dectrain\\labels_decrec\\UAVSig%d_DA%d.txt', i, m), writeMark);
    %                 fid_snr = fopen(sprintf('D:\\download\\UAVDataset\\dectrain\\labels_snr\\UAVSig%d_DA%d.txt', i, m), writeMark);
                elseif i > round(numSample/10*6) && i <= round(numSample/10*8)  
                    continue
    %                 fid_dec = fopen(sprintf('D:\\download\\UAVDataset\\decvalid\\labels_dec\\UAVSig%d_DA%d.txt', i, m), writeMark);
    %                 fid_decrec = fopen(sprintf('D:\\download\\UAVDataset\\decvalid\\labels_decrec\\UAVSig%d_DA%d.txt', i, m), writeMark);
    %                 fid_snr = fopen(sprintf('D:\\download\\UAVDataset\\decvalid\\labels_snr\\UAVSig%d_DA%d.txt', i, m), writeMark);
                else
                    fid_dec_snr = fopen(sprintf('D:\\download\\UAVDataset\\dectest_fixsnr\\labels_dec_snr\\UAVSig%d_DA%d.txt', round(numSample/10*8)+5*(i-round(numSample/10*8)-1)+ak, m), writeMark);
                    fid_decrec_snr = fopen(sprintf('D:\\download\\UAVDataset\\dectest_fixsnr\\labels_decrec_snr\\UAVSig%d_DA%d.txt', round(numSample/10*8)+5*(i-round(numSample/10*8)-1)+ak, m), writeMark);
                end
                if i <= round(numSample/10*8)
                     continue
    %                 fprintf(fid_dec, '%d %.6f %.6f %.6f %.6f\n', 0, x_center(k), y_center(k), width(k), height(k)); 
    %                 fprintf(fid_decrec, '%d %.6f %.6f %.6f %.6f\n', labellist(k), x_center(k), y_center(k), width(k), height(k)); 
    %                 fprintf(fid_snr, '%d %.6f %.6f %.6f %.6f %d\n', labellist(k), x_center(k), y_center(k), width(k), height(k), snrlist(k)); 
                else
                    if k == ak
                        fprintf(fid_dec_snr, '%d %.6f %.6f %.6f %.6f %d %d\n', 0, x_center(k), y_center(k), width(k), height(k), snrlist(k), 1); 
                        fprintf(fid_decrec_snr, '%d %.6f %.6f %.6f %.6f %d %d\n', labellist(k), x_center(k), y_center(k), width(k), height(k), snrlist(k), 1);                         
                    else
                        fprintf(fid_dec_snr, '%d %.6f %.6f %.6f %.6f %d\n', 0, x_center(k), y_center(k), width(k), height(k), snrlist(k)); 
                        fprintf(fid_decrec_snr, '%d %.6f %.6f %.6f %.6f %d\n', labellist(k), x_center(k), y_center(k), width(k), height(k), snrlist(k)); 
                    end
                end
                if i <= round(numSample/10*8)
                    fclose(fid_dec); 
                    fclose(fid_decrec); 
                    fclose(fid_snr);
                else
                    fclose(fid_dec_snr); 
                    fclose(fid_decrec_snr);
                end
            end
    
            SigSetArray = double(cell2mat(SigSet));
            noise = (randn(1,size(SigSetArray, 2)) + 1i*randn(1,size(SigSetArray, 2))) / sqrt(2); % 产生1W的噪声
            SigSetArray = [SigSetArray; noise];
            FinalSig = sum(SigSetArray, 1); % 合并单一信号
            % 保存yolo时频灰度图!!!!!!
            [f, t, P] = abstft(FinalSig, fs, 2048, 0);
            PP = minmax(P) * 255;
            im = mat2gray(PP);
            if i <= round(numSample/10*6)
%                 imwrite(im, sprintf('D:\\download\\UAVDataset\\dectrain\\images\\UAVSig%d_DA%d.png', i, m));
            elseif i > round(numSample/10*6) && i <= round(numSample/10*8)  
%                 imwrite(im, sprintf('D:\\download\\UAVDataset\\decvalid\\images\\UAVSig%d_DA%d.png', i, m));
            else
                imwrite(im, sprintf('D:\\download\\UAVDataset\\dectest_fixsnr\\images\\UAVSig%d_DA%d.png', round(numSample/10*8)+5*(i-round(numSample/10*8)-1)+ak, m));
            end
        %     imshow(im);  % 显示图像
        %     title('灰度图像'); 
        %     DatasetInfos(i).selectedDatasets{k};
        
%             for k = 1:numPerSig % 下变频滤波，保存信号时域信息
%                 if DA_freshift == true
%                     da_freshiftnum = 4;
%                 else
%                     da_freshiftnum = 1;
%                 end
%                 for j=1:da_freshiftnum % 下变频滤波，数据增强
%                     if offset == true
%                         bwoffset = randi([-20,20]);
%                         freqoffset = randi([-20,20]);
%                         spoffset = randi([0,2]);
%                         epoffset = randi([0,2]);
%                     else
%                         [bwoffset, freqoffset, spoffset, epoffset] = deal(0, 0, 0, 0);
%                     end
%                     ccos = cos(2 * pi * (-fclist(k)+freqoffset*fs/20480) / fs * (0:length(FinalSig)-1));
%                     csin = sin(2 * pi * (-fclist(k)+freqoffset*fs/20480) / fs * (0:length(FinalSig)-1));
%                     rsig = bl_filter(FinalSig .* (ccos + 1i * csin), (bwlist(k)+bwoffset*fs/20480)/2, fs, 8);
%                     % [f, t, P] = tstft(rsig, fs, 2048, 512);
%                     SepSig = rsig(startpoint(k)+spoffset*2048:endpoint(k)-epoffset*2048);
%                     % 保存单信号时频图
%                     [f, t, P] = abstft(SepSig, fs, 2048, 0);
%                     % B = Bispectrum(SepSig, 2048);
%                     PP = minmax(P) * 255;
%                     im = mat2gray(PP);
%                     height = size(im, 1);
%                     width = size(im, 2);
%                     if height > 1536 % 确保高度大于 1024，以便裁剪上下各 768 像素
%                         sub = round(2048*(1-bwlist(k)/fs)/2);
%                         im_cropped = im(sub+1:height-sub, :); % 裁剪上下各 768 像素
%                     else
%                         error('图像高度不足以裁剪。');
%                     end
% %                     if i <= round(numSample/10*6)
% %                         imwrite(im_cropped, sprintf('D:\\download\\UAVDataset\\rectrain\\OneUAVImage%d_DA%d-%d_%d_snr%d_DA%d.png', i, m, k, labellist(k), snrlist(k), j));
% %                     elseif i > round(numSample/10*6) && i <= round(numSample/10*8)  
% %                         imwrite(im_cropped, sprintf('D:\\download\\UAVDataset\\recvalid\\OneUAVImage%d_DA%d-%d_%d_snr%d_DA%d.png', i, m, k, labellist(k), snrlist(k), j));
% %                     else
% %                         imwrite(im_cropped, sprintf('D:\\download\\UAVDataset\\rectest\\OneUAVImage%d_DA%d-%d_%d_snr%d_DA%d.png', i, m, k, labellist(k), snrlist(k), j));
% %                     end
%     
%     %% 保存子yolo标签!!!!!!
%                     [currentHeight, currentWidth, ~] = size(im_cropped);
%                     targetHeight = 512;
%                     targetWidth = 512;
%                     padTop = floor((targetHeight - currentHeight) / 2);
%                     padBottom = ceil((targetHeight - currentHeight) / 2);
%                     padLeft = floor((targetWidth - currentWidth) / 2);
%                     padRight = ceil((targetWidth - currentWidth) / 2);
%                     im_yolocropped = padarray(im_cropped, [padTop, padLeft], 0, 'pre');
%                     im_yolocropped = padarray(im_yolocropped, [padBottom, padRight], 0, 'post');
%                     if i <= round(numSample/10*6)
%                         fid = fopen(sprintf('D:\\download\\UAVDataset\\rectrain_yolo\\labels\\OneUAVImage%d_DA%d-%d_%d_snr%d_DA%d.txt', i, m, k, labellist(k), snrlist(k), j), writeMark);
%                         imwrite(im_yolocropped, sprintf('D:\\download\\UAVDataset\\rectrain_yolo\\images\\OneUAVImage%d_DA%d-%d_%d_snr%d_DA%d.png', i, m, k, labellist(k), snrlist(k), j));
%                         fclose(fid); 
%                     elseif i > round(numSample/10*6) && i <= round(numSample/10*8)  
%                         fid = fopen(sprintf('D:\\download\\UAVDataset\\recvalid_yolo\\labels\\OneUAVImage%d_DA%d-%d_%d_snr%d_DA%d.txt', i, m, k, labellist(k), snrlist(k), j), writeMark);
%                         imwrite(im_yolocropped, sprintf('D:\\download\\UAVDataset\\recvalid_yolo\\images\\OneUAVImage%d_DA%d-%d_%d_snr%d_DA%d.png', i, m, k, labellist(k), snrlist(k), j));
%                         fclose(fid); 
%                     end
%                     if i <= round(numSample/10*8)
%                         fprintf(fid, '%d %.6f %.6f %.6f %.6f\n', labellist(k), 0.5, 0.5, width/512, (height-2*sub)/512); 
%                         fclose(fid); 
%                     end
%     
%     %% 保存其它特征    
%         %             % 保存单信号时域特征!!!!!!
%                     feat = features(SepSig); % Feature extraction
%                     if i <= round(numSample/10*6)
%                         save(sprintf('D:\\download\\UAVDataset\\rectrain\\OneUAVFeat%d_DA%d-%d_%d_snr%d_DA%d.mat', i, m, k, labellist(k), snrlist(k), j), 'feat');
%                     elseif i > round(numSample/10*6) && i <= round(numSample/10*8)  
%                         save(sprintf('D:\\download\\UAVDataset\\recvalid\\OneUAVFeat%d_DA%d-%d_%d_snr%d_DA%d.mat', i, m, k, labellist(k), snrlist(k), j), 'feat');
%                     end
%     %                 else
%     %                     save(sprintf('D:\\download\\UAVDataset\\rectest\\OneUAVFeat%d_DA%d-%d_%d_snr%d_DA%d.mat', i, m, k, labellist(k), snrlist(k), j), 'feat');
%     %                 end
%     
%         %             SepSig = single(SepSig);                   
%         %             % 保存单信号IQ序列!!!!!!
%         %             save(sprintf('D:\\download\\UAV-Sigmf-float16\\UAVsigmf-npdata\\rectest\\OneUAVIQ%d_%d_burst%d_%d_%d.mat', uavnum, snr, burstnum, num, i), 'SepSig');
%                     fprintf('生成了OneUAVImage%d_DA%d-%d_%d_snr%d_DA%d.png\n', i, m, k, labellist(k), snrlist(k), j);
%                 end
%             end
        %             figure;
        %             freq_spectrum = fft(FinalSig);
        %             amp_spectrum = abs(freq_spectrum);
        %             plot(1:length(FinalSig), amp_spectrum);
        %             fprintf('Complex carrier signal shape: %d\n', length(FinalSig));
            % 保存接收信号IQ序列（test!）!!!!!!
        %    FinalSig = single(FinalSig);
            if i > round(numSample/10*8)
                save(sprintf('D:\\download\\UAVDataset\\alltest_fixsnr\\UAVSig%d_DA%d.mat', round(numSample/10*8)+5*(i-round(numSample/10*8)-1)+ak, m), 'FinalSig');
            end
        end
    end
end

fclose('all'); % 关闭内部打开的所有文件




% Import data function
function orisig = origininput(filename)
    % Read JSON metadata (if needed)
    meta_dict = jsondecode(fileread(strcat(filename, '.json')));    
    % Open binary file
    fid = fopen(strcat(filename, '.bin'), 'rb');    
    % Read data as uint16
    raw_data = fread(fid, 'uint16');    
    % Close the file
    fclose(fid);   
    % Convert to single precision
    orisig = halfPrecisionToSingle(raw_data);
end

function singleData = halfPrecisionToSingle(halfData)
    % Convert half-precision (16-bit) data to single precision    
    % Extract sign, exponent, and mantissa
    sign = bitshift(bitand(halfData, 32768), -15);
    exponent = bitshift(bitand(halfData, 31744), -10);
    mantissa = bitand(halfData, 1023);    
    % Adjust exponent and mantissa for single precision
    singleExponent = exponent - 15 + 127;
    singleMantissa = bitshift(mantissa, 13);    
    % Handle special cases
    singleExponent(exponent == 0) = 0;
    singleMantissa(exponent == 0) = bitshift(mantissa(exponent == 0), 13);
    singleExponent(exponent == 31) = 255;    
    % Combine into single precision
    singleBits = bitor(bitshift(sign, 31), bitor(bitshift(singleExponent, 23), singleMantissa));
    singleData = typecast(uint32(singleBits), 'single');
end

% Upsampling and complex signal creation
function data0 = upsampleSig(orisig)
    even = orisig(1:2:end);
    odd = orisig(2:2:end);
    evenx = linspace(0, length(even) * 10 - 10, length(even) * 10 - 9);
    evenxp = linspace(0, length(even) * 10 - 10, length(even));
    upeven = interp1(evenxp, even, evenx, 'spline');
    oddx = linspace(0, length(odd) * 10 - 10, length(odd) * 10 - 9);
    oddxp = linspace(0, length(odd) * 10 - 10, length(odd));
    upodd = interp1(oddxp, odd, oddx, 'spline');    
    data0 = upeven + 1i * upodd;
end

% Butterworth lowpass filter
function y = bl_filter(data, cutoff, fs, order)
    [b, a] = butter(order, cutoff / (0.5 * fs), 'low');
    y = filter(b, a, data);
end

% Frequency shift
function [data, fc] = rf_modu(data0, bw, j)
    global fs fc0;
    channel = [-4*fc0, -2*fc0, 0, 2*fc0, 4*fc0];
    fc = channel(j)+randi([-(2*fc0/2-bw/2),(2*fc0/2-bw/2)]);
    ccos = cos(2 * pi * fc / fs * (0:length(data0)-1));
    csin = sin(2 * pi * fc / fs * (0:length(data0)-1));
    data = data0 .* (ccos + 1i * csin);
end

% Power modify
function x = ChangePower(x, snr)
    Xpower = sum(abs(x - mean(x)).^2) / length(x);
    snrLinear = 10^(snr / 10);
    x = x * sqrt(snrLinear / Xpower);
%     noise = (randn(size(x)) + 1i*randn(size(x))) / sqrt(2); % 产生1W的功率
end

% Short-Time Fourier Transform (STFT)
function [f, t, P] = abstft(data1, fs, nperseg, overlap)
    [S, f, t] = stft(data1, fs, Window=hamming(nperseg), OverlapLength = overlap, FFTLength = nperseg);
    P = abs(S);
    global showfig
    if showfig == 1
%         figure;
%         % surf(t, f, 10*log10(P), 'EdgeColor', 'none');
%         imagesc(t, f, P);
%         axis xy; axis tight; %colormap(jet); %view(0, 90);
%         xlabel('时间 (s)');
%         ylabel('频率(Hz)');
%         title('时频图');
        figure;
        % 将幅度转换为 dB
        P_dB = 10*log10(P);
        imagesc(t, f, P_dB);
        axis xy; 
        axis tight;
        colormap('parula'); % 蓝黄配色
        colorbar; % 显示幅度条
        xlabel('时间 (s)');
        ylabel('频率 (Hz)');
        title('时频图 (dB)');
    end
end

function B = Bispectrum(x, nfft)
    % x: 输入信号
    % nfft: FFT 点数
    % 确保输入信号为列向量
    x = x(:);
    % 计算信号的长度
    N = length(x);
    % 计算信号的傅里叶变换
    X = fft(x, nfft);
    % 初始化双谱矩阵
    B = zeros(nfft, nfft);
    % 计算双谱
    for f1 = 1:nfft
        for f2 = 1:nfft
            % 计算双谱的每个分量
            B(f1, f2) = mean(X(f1) * X(f2) * conj(X(mod(f1+f2-1, nfft) + 1)));
        end
    end
    B = abs(B);
    global showfig
    if showfig == 1
        % 显示双谱
        figure;
        imagesc(B);
        title('双谱幅值');
        xlabel('频率 f1');
        ylabel('频率 f2');
        colorbar;
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
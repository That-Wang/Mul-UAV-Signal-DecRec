function SigOfdm = GenOfdm(fs,bw,len)
% 参数设置
% fs = 80e6; % 采样率 100 MHz
% bw = 10e6; % 信号带宽 10 MHz
% len = 8e5; % 信号总采样点数
% 参数设置
sps = round(fs / bw); % 每符号采样点数
numSubcarriers = 64; % 子载波数量
cpLen = 16; % 循环前缀长度

% 根据调制类型设置调制阶数
% 随机选择调制类型
modTypes = {'QPSK', '16QAM'};
modType = modTypes{randi([1, 2])};
switch modType
    case 'QPSK'
        modOrder = 4;
    case '16QAM'
        modOrder = 16;
    otherwise
        error('Unsupported modulation type. Use ''QPSK'' or ''16QAM''.');
end

numSymbols = ceil(len / sps / (numSubcarriers + cpLen)); % 符号数量

% 生成随机比特流
numBits = numSubcarriers * numSymbols * log2(modOrder);
bitStream = randi([0 1], numBits, 1);

% 调制
symbols = qammod(bitStream, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);

% 将符号分成 OFDM 符号
symbols = reshape(symbols, numSubcarriers, numSymbols);

% IFFT 以生成时间域信号
timeDomainSymbols = ifft(symbols, numSubcarriers);

% 添加循环前缀
cyclicPrefix = timeDomainSymbols(end-cpLen+1:end, :);
timeDomainSymbolsWithCP = [cyclicPrefix; timeDomainSymbols];

% 将所有符号连接成一个信号
txSignal = timeDomainSymbolsWithCP(:);

% 确保信号长度为 len
txSignal = txSignal(1:min(end, round(len/sps))); % 防止超出

% 插值以匹配采样率
txSignalUpsampled = upsample(txSignal, sps);

% 设计升余弦滤波器
rolloff = 0.25; % 升余弦滚降系数
span = 10; % 滤波器跨度（符号数）
rcFilter = rcosdesign(rolloff, span, sps, 'normal');

% 对信号进行升余弦滤波
SigOfdm = conv(txSignalUpsampled, rcFilter, 'same');

% % 绘制信号时域波形
% t = (0:length(SigOfdm)-1) / fs;
% figure;
% plot(t, real(SigOfdm));
% title('Filtered OFDM Signal (Real Part)');
% xlabel('Time (s)');
% ylabel('Amplitude');
% 
% % 绘制信号频谱
% figure;
% pwelch(SigOfdm, 1024, [], [], fs, 'centered');
% title('Power Spectral Density of Filtered OFDM Signal');
end


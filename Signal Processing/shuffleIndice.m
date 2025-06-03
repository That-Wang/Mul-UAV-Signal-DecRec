function shuffledIndiceall = shuffleIndice(numPerGroup,datasetNames,totalData,moreOFDM)
% 参数设置
% numPerGroup = 5; % 每组的数量
% numDatasets = 10; % 从 0 到 9，共 10 个数字
% totalData = 100; % 每个数字需要被选 100 次

% 初始化计数器和结果数组
numDatasets = numel(datasetNames);
shuffledIndiceall = [];
if moreOFDM == true
    datasetNames = [datasetNames, 'OFDM2'];
    numDatasets = numel(datasetNames);
end
labels = [];
for i = 1:numDatasets
    switch datasetNames{i}
        case 'DJI M100(1)'
            labels(i) = 0;
        case 'DJI M100(2)'
            labels(i) = 1;
        case 'DJI M100(3)'
            labels(i) = 2;
        case 'DJI M100(4)'
            labels(i) = 3;
        case 'DJI M100(5)'
            labels(i) = 4;
        case 'OFDM'
            labels(i) = 5;
        case 'DJI Inspire2'
            labels(i) = 6;
        case 'DJI mini2'
            labels(i) = 7;
        case 'DJI Matrice pro'
            labels(i) = 8;
        case 'DJI Mavic'
            labels(i) = 9;
        case 'OFDM2'
            labels(i) = 5;
    end
end

% 生成索引
datanums = [round(totalData/10*6), round(totalData/10*2), totalData-round(totalData/10*6)-round(totalData/10*2)];
for i = 1:3
    counts = zeros(1, numDatasets);
    shuffledIndices = [];
    while any(counts < datanums(i))
        % 找到所有未满 totalData 的数字
        availableNumbers = find(counts < datanums(i)) - 1;
        
        % 如果剩余的可用数字正好是 numPerGroup 的倍数
        if length(availableNumbers) >= numPerGroup
            selectedNumbers = randsample(availableNumbers, numPerGroup, false);
        else
            % 如果不是，直接退出循环（理想情况下不会发生）
            break;
        end
        
        % 更新结果和计数器
        shuffledIndices = [shuffledIndices, selectedNumbers];
        for num = selectedNumbers
            counts(num + 1) = counts(num + 1) + 1;
        end
    end
    
    % 确保所有数字都被分配完
    remainingNumbers = [];
    for num = 0:numDatasets-1
        remainingNumbers = [remainingNumbers, repmat(num, 1, datanums(i) - counts(num + 1))];
    end
    
    % 将剩余数字按 numPerGroup 为一组随机分配
    while ~isempty(remainingNumbers)
        if length(remainingNumbers) >= numPerGroup
            selectedNumbers = randsample(remainingNumbers, numPerGroup, false);
        else
            selectedNumbers = remainingNumbers;
        end
        shuffledIndices = [shuffledIndices, selectedNumbers];
        % 删除每个已选择的数字中的一个实例
        for num = selectedNumbers
            idx = find(remainingNumbers == num, 1, 'first');
            remainingNumbers(idx) = [];
        end
    end
    
    % 找到唯一的数字并排序
    uniqueNumbers = unique(shuffledIndices);
    % 创建映射，将唯一数字映射到新的值
    numberToValueMap = containers.Map(uniqueNumbers,labels);
    % 替换原数组中的数字
    newShuffledIndices = arrayfun(@(num) numberToValueMap(num), shuffledIndices);
    if mod(datanums(i)*numDatasets, numPerGroup) 
       newShuffledIndices = [newShuffledIndices, repmat(5, 1, numPerGroup-mod(datanums(i)*numDatasets, numPerGroup))]; % 5代表OFDM    
    end
    shuffledIndices = newShuffledIndices+1;
    uniqueValues = unique(shuffledIndices);
    countss = accumarray(shuffledIndices', 1);
    for k = 1:length(uniqueValues)
        fprintf('Number %d: %d times\t', uniqueValues(k), countss(uniqueValues(k)));
    end
    fprintf('\n');
    shuffledIndiceall = [shuffledIndiceall, shuffledIndices];
end

end


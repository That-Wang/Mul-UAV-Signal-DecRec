function result = selectRandomDatasets(datasetNames, numPerGroup, totalSig, moreOFDM)
    % Define dataset names
%     numPerGroup = 5;
    datasetNamesAll = {'DJI M100(1)', 'DJI M100(2)', 'DJI M100(3)', ...
                    'DJI M100(4)', 'DJI M100(5)', 'OFDM', 'DJI Inspire2', ...
                    'DJI mini2', 'DJI Matrice pro', 'DJI Mavic'};
    numDatasets = numel(datasetNamesAll);
    totalData = totalSig;
    % 生成均匀随机索引
    shuffledIndices = shuffleIndice(numPerGroup,datasetNames,totalData,moreOFDM);

    % when test
%     shuffledIndices(1:5) = [1,2,10,9,6];
    
    % Initialize result structure
    result = struct('selectedDataNames', {}, 'folderPaths', {}, ...
                    'nums', {}, 'filenames', {}, 'labels', {}, 'bws', {});
    
    % Initialize counters for each dataset
    counters = zeros(1, numDatasets);
    
    % Loop through and select groups of datasets
    for i = 1:numPerGroup:numel(shuffledIndices)
        groupIndices = shuffledIndices(i:min(i+numPerGroup-1, numel(shuffledIndices)));
        selectedDataNames = datasetNamesAll(groupIndices);
        
        folderPaths = cell(1, numel(groupIndices));
        labels = cell(1, numel(groupIndices));
        bws = zeros(1, numel(groupIndices));
        nums = zeros(1, numel(groupIndices));
        filenames = cell(1, numel(groupIndices));
        
        for j = 1:numel(groupIndices)
            index = groupIndices(j);
            [folderPaths{j}, labels{j}, bws(j)] = getDatasetInfo(datasetNamesAll{index});
            
            % Increment the counter for the selected dataset
            counters(index) = counters(index) + 1;
            nums(j) = counters(index);
            
            % Construct the filename using the dataset name
            datasetName = transName(selectedDataNames{j});
            switch datasetName
                case {'uav1_6ft_burst1','uav2_6ft_burst1','uav3_6ft_burst1','uav4_6ft_burst1','uav5_6ft_burst1','OFDM'}
                    filename = sprintf('%s%s_%d', folderPaths{j}, datasetName, nums(j));
                case {'inspire2','mini2','matrice','mavic'}
                    filename = sprintf('%s%s-%d', folderPaths{j}, datasetName, nums(j));
            end
            filenames{j} = filename;
        end
        
        result(end+1).selectedDataNames = selectedDataNames;
        result(end).folderPaths = folderPaths;
        result(end).labels = labels;
        result(end).bws = bws;
        result(end).nums = nums;
        result(end).filenames = filenames;
    end
end

function [folderPath, label, bw] = getDatasetInfo(selectedDataset)
    switch selectedDataset
        case 'DJI M100(1)'
            folderPath = 'D:\\download\\UAV-Sigmf-float16\\'; 
            label = '0';
            bw = 9.6e6;
        case 'DJI M100(2)'
            folderPath = 'D:\\download\\UAV-Sigmf-float16\\'; 
            label = '1';
            bw = 9.6e6;
        case 'DJI M100(3)'
            folderPath = 'D:\\download\\UAV-Sigmf-float16\\'; 
            label = '2';
            bw = 9.6e6;
        case 'DJI M100(4)'
            folderPath = 'D:\\download\\UAV-Sigmf-float16\\'; 
            label = '3';
            bw = 9.6e6;
        case 'DJI M100(5)'
            folderPath = 'D:\\download\\UAV-Sigmf-float16\\'; 
            label = '4';
            bw = 9.6e6;
        case 'OFDM'
            folderPath = 'D:\\download\\UAV-Sigmf-float16\\'; 
            label = '5';
            bw = randsample([18.2e6, 9.6e6], 1); % 10e6; % 20e6, 10e6
        case 'DJI Inspire2'
            folderPath = 'D:\\download\\UAVDataset\\inspire2Vid\\'; 
            label = '6';
            bw = 9.6e6;
        case 'DJI mini2'
            folderPath = 'D:\\download\\UAVDataset\\mini2vid\\'; 
            label = '7';
            bw = 18.2e6;
        case 'DJI Matrice pro'
            folderPath = 'D:\\download\\UAVDataset\\matricevid\\'; 
            label = '8';
            bw = 18.2e6;
        case 'DJI Mavic'
            folderPath = 'D:\\download\\UAVDataset\\mavicVid1\\'; 
            label = '9';
            bw = 18.2e6;
    end
end

function datasetName = transName(selectedDataset)
    switch selectedDataset
        case 'DJI M100(1)'
            datasetName = 'uav1_6ft_burst1';
        case 'DJI M100(2)'
            datasetName = 'uav2_6ft_burst1';
        case 'DJI M100(3)'
            datasetName = 'uav3_6ft_burst1';
        case 'DJI M100(4)'
            datasetName = 'uav4_6ft_burst1';
        case 'DJI M100(5)'
            datasetName = 'uav5_6ft_burst1';
        case 'OFDM'
            datasetName = 'OFDM';
        case 'DJI Inspire2'
            datasetName = 'inspire2';
        case 'DJI mini2'
            datasetName = 'mini2';
        case 'DJI Matrice pro'
            datasetName = 'matrice';
        case 'DJI Mavic'
            datasetName = 'mavic';
    end
end



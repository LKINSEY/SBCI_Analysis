
function summarizeSLAP2_slowIndicators(dr)
%call up a GUI for the user to define Soma ROI and regions to exclude
% base = 'Y:\scratch\ophys\Lucas\BCI_gCaMP6s\slap2';
% choice = { 
%     'slap2_760267_2024-11-05_15-57-50\Neuron1',
%     'slap2_760267_2024-11-07_14-43-10\neurons_plural',
%     'slap2_760267_2024-11-08_16-10-35\neurons_plural'
%     };
% dr = [base filesep choice{3}];
disp(['Looking at session:  ' dr])


trials = dir([dr filesep '*_REGISTERED_DOWNSAMPLED-80Hz.tif']);
bciEpoch = 2;

firstTrialTiffs = {
[trials(1).folder filesep trials(1).name],
[trials(2).folder filesep trials(2).name]
}

nDMDs = 2; 
fnAnn = [dr filesep 'ANNOTATIONS.mat'];
if exist(fnAnn, 'file')
    load(fnAnn, 'ROIs')
    disp('Loading Annotations that were already made')
else
    for DMDix = 1:nDMDs
        trialStr = firstTrialTiffs{DMDix};
        IM = copyReadDeleteScanImageTiff(trialStr);
        IM = squeeze(mean(IM,[3 4], 'omitnan'));
        hROIs(DMDix) = drawROIs(sqrt(max(0,IM)), trials(DMDix).folder , trials(DMDix).name);  %soma0 should always be the BCI neuron
        ROIs(DMDix).dr = trials(DMDix).folder;
        ROIs(DMDix).fn = trials(DMDix).name;
    end
    for DMDix = 1:nDMDs
        waitfor(hROIs(DMDix).hF);
        ROIs(DMDix).roiData = hROIs(DMDix).roiData; %to call a cell's roi data, call "ROIs(DMDix).roiData(CELLix)"
        
    end
    save(fnAnn, 'ROIs'); clear hROIs;
end

if exist([dr filesep 'trialTable.mat'], 'file')
    trialTable = load([dr filesep 'trialTable.mat']).trialTable;
else
    disp('NO TRIAL TABLE DETECTED BURN EVERYTHING AND GO HOME!!!!!')
    return
end


numChannels = 2;
for trialIX = 1:length(trialTable.epoch)
    for dmdIX = 1:nDMDs
    
        epoch = trialTable.epoch(trialIX);
        IMfn = [dr filesep 'E' num2str(epoch) 'T' num2str(trialIX) 'DMD' num2str(dmdIX) '_REGISTERED_DOWNSAMPLED-80Hz.tif'];
        
        IMtiff = copyReadDeleteScanImageTiff(IMfn);
        IMtiff = reshape(IM, size(IM,1), size(IM,2), numChannels, []); %deinterleave
        
        channel1_mean = normalizeLocal(mean(squeeze(IMtiff(:,:,1,:)), 3, 'omitnan'), 21, 0.9, false);
        channel2_mean = normalizeLocal(mean(squeeze(IMtiff(:,:,2,:)), 3, 'omitnan'), 21, 0.9, false);
        expmtSummary{dmdIX, trialIX}.meanIM = reshape([channel1_mean channel2_mean], [size(channel1_mean,1) size(channel1_mean,2) 2]); %save mean im of fov for both channesl

        channel1_tiff = squeeze(IMtiff(:,:,1,:)); %gCaMP6s
        channel2_tiff = squeeze(IMtiff(:,:,2,:)); %tdTomato

        for somaIX = 1:length(ROIs(dmdIX).roiData)

            somaFootprint = ROIs(dmdIX).roiData{1,somaIX}.mask;
            somaID = ROIs(dmdIX).roiData{1,somaIX}.Label;
            tiffmask = repmat(somaFootprint, [1,1,size(channel1_tiff, 3)]);
            somaTrace = tiffmask .* channel1_tiff;
            meanTrace = squeeze(mean(somaTrace, [1 2], 'omitnan'));
            somaCell = {
                somaID,
                meanTrace};
            expmtSummary{dmdIX, trialIX}.roiData{somaIX} =somaCell;
        end


    end
end

%save the data
savedr = [dr filesep 'ExperimentSummary'];
if ~exist(savedr, 'dir')
    mkdir(savedr);
end
fnsave = [savedr filesep 'Summary-' datestr(now, 'YYmmDD-HHMMSS') '.mat'];
save(fnsave, 'exptSummary', "-v7.3");
disp('DoneSummarizing Slow Blinking Neurons')
end


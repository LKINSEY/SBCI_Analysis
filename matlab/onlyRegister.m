function onlyRegister(dr)

if ~nargin
    dr = uigetdir; %neuron folder where scans are, not project folder
end

%generate the trial table
if ~exist([dr filesep 'trialTable.mat'], 'file')
    buildTrialTable(dr);
end

%align files
gCaMP6sOnlyReg(dr);%extremely liberal registration process with no drapping of trials

%generate file that has soma activity
summarizeSLAP2_slowIndicators(dr);
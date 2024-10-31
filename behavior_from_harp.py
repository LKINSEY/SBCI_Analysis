#%% #IMPORTS
#IMPORTS
import numpy as np
from scipy.signal import savgol_filter
from mat73 import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
bks = matplotlib.rcsetup.interactive_bk
plt.switch_backend('WebAgg')
from behaviorUtil import *
#%% INITIALIZATION
# INIT (for single session use)
#will eventually format for general session analysis

dataDir = 'Z:/ophys/Lucas/Agmat/Experiments/behavior/20241030T173008_neuron2/'
exmptPath = 'Z:/ophys/Lucas/Agmat/Experiments/slap2/slap2_748313_2024-10-30_16-09-27/'#/Neuron2/ExperimentSummary/'
neuronNumber = 2
sessionData = load_behavior_df(dataDir)
trials = trializeData(sessionData)

exmptSummaryPath = exmptPath + f'Neuron{neuronNumber}/ExperimentSummary/Summary-241030-194541.mat'
expmtSummary = loadmat(exmptSummaryPath)

# %%

neuronNumber = 2
mouseID = '748313'
performance = []
rxnTimes = []
motorADCCompare = []
for trial in trials.keys():
    #for plotting learning
    performance.append( trials[trial]['Outcome']   )
    rxnTimes.append(trials[trial]['deltaT'])

    #for plotting motor and ADC
    adc = trials[trial]['ADC']
    motor = trials[trial]['Motor']
    spacer = np.empty((1,len(adc))); spacer[:] = np.nan
    motorADCCompare.append([adc, motor, spacer])



#learning curve
performance = np.array(performance); rxnTimes = np.array(rxnTimes)
perfKernel = np.ones(10)/10 *100
# smoothedPerformance = np.convolve(performance, perfKernel, mode='same')
smoothedPerformance = savgol_filter(performance, window_length=10, polyorder=1)

## PLOTS
plt.close('all')
plt.figure(0)
fig, ax1 = plt.subplots()
ax1.plot(smoothedPerformance, 'g-')
ax1.set_ylabel(f'Performance Curve for neuron {neuronNumber}', color='g')
ax2 = ax1.twinx()
ax2.scatter(range(len(rxnTimes)),rxnTimes, color='b')
ax2.set_ylabel(f'ReactionTimes of Neuron {neuronNumber}', color='b')
plt.title(f'Behaviorn Performance of Neuron {neuronNumber} for {mouseID}')
plt.show()

#%%
plt.close('all')
fig2, ax = plt.subplots()
cmaps = ['winter', 'autumn', 'bone'] #for motor and adc, respectively
rowCount = 0
for row in motorADCCompare:
    trialADC = row[0].values.reshape(1,-1)
    trialMotor = row[1].values.reshape(1,-1)
    trialSpacer = row[2].reshape(1,-1)
    trialData = np.vstack((trialADC, trialMotor, trialSpacer))
    adcSHOW = ax.imshow(trialData[0:1], cmap='winter', aspect='auto', extent=[0, trialData.shape[1], rowCount,rowCount + 1])
    motorSHOW = ax.imshow(trialData[1:2], cmap='autumn', aspect='auto', extent=[0, trialData.shape[1],rowCount + 1,rowCount + 2])
    spacerShow = ax.imshow(trialData[2:3], cmap='bone', aspect='auto', extent=[0, trialData.shape[1], rowCount + 2,rowCount + 3])
    rowCount += 3

maxRow = max(len(trial[0]) for trial in motorADCCompare)
ax.set_xlim(0,maxRow)
ax.set_ylim(0, 3*len(motorADCCompare))

cbar1 = plt.colorbar(adcSHOW, ax=ax, orientation='vertical', pad = 0.05)
cbar1.set_label('BCI Analog Input')
cbar2 = plt.colorbar(motorSHOW, ax=ax, orientation='vertical', pad = 0.15)
cbar2.set_label('Motor Commands')

ax.set_yticks([])

plt.title('BCI Analog Input Compared to Motor Commands')
plt.xlabel('Trial time')
plt.ylabel('Trial')

plt.show()
#%% 2P DATA ANALYSIS
## 2P DATA ANALYSIS

exmptSummaryPath = exmptPath + f'Neuron{neuronNumber}/ExperimentSummary/Summary-241030-194541.mat'
expmtSummary = loadmat(exmptSummaryPath)

# %%

#check which dmd has soma
somaLoc = []
for i in range(len(expmtSummary['exptSummary']['userROIs'])):
    somaLoc.append(len(expmtSummary['exptSummary']['userROIs'][i]))
somaLoc = np.where(np.array(somaLoc))[0][0]

dmd1_byTrial = np.array(expmtSummary['exptSummary']['E'])[:,0]
dmd2_byTrial = np.array(expmtSummary['exptSummary']['E'])[:,1]
somaDmd_byTrial = np.array(expmtSummary['exptSummary']['E'])[:,somaLoc]
caTrace = []

#get calcium stuff
for trial in somaDmd_byTrial:
    try:
        trace =np.array(trial['ROIs']['F'][:,1])
        noNaNs_trace = trace[~np.isnan(trace)]
        m, b, _, _, _ =  stats.linregress(range(len(noNaNs_trace)), noNaNs_trace)
        baseline_linear = (m*np.arange(0, len(trace), 1)) + b
        result = trace-baseline_linear
        caTrace.append(np.nanmean(np.array(result)))
        print(np.nanmean(result))
    
    except Exception:
        caTrace.append(np.NaN)
        print('NaN')
calcium_byTrial = np.array(caTrace)

#%%

#get glutamate stuff
dmd1Synapses = []
dmd2Synapses = []
firstTrial = True

for trial in dmd1_byTrial:
    try:
        firstTrial = False
        glutTraces = np.array(trial['denoised'])
        synapticActivity_byTrial = np.empty(glutTraces.shape)
        nSynapses = glutTraces.shape[0]
        #the long way to remove bleaching for now... :(
        for synapseIDX in range(glutTraces.shape[0]):
            noNaNs_synapse = glutTraces[synapseIDX, ~np.isnan(glutTraces[synapseIDX,:])]
            m, b, _, _, _ = stats.linregress(range(len(noNaNs_synapse)), noNaNs_synapse)
            baseline_synapse = (m*np.arange(0, len(glutTraces[synapseIDX,:]), 1)) + b
            synapseActivity = glutTraces[synapseIDX, :] - baseline_synapse
            synapticActivity_byTrial[synapseIDX,:] = synapseActivity
            
        averagePerSynapse_byTrial = np.reshape(np.nanmean(synapticActivity_byTrial, axis=1), (nSynapses,1))
        dmd1Synapses.append(averagePerSynapse_byTrial)
        print(averagePerSynapse_byTrial.shape)

    except Exception:
        if firstTrial:
            firstTrial = False
            pass 
        nanSynapses = np.empty((nSynapses,1))
        nanSynapses[:] = np.nan
        dmd1Synapses.append(nanSynapses)
        print('NaN', nanSynapses.shape)

#glut averages by trial for each synapse
dmd1Synapses = np.hstack(dmd1Synapses)

#find increase or decrease in activity
dmd1SynapseLearningCurves = []
for synapseIDX in range(dmd1Synapses.shape[0]):
    glut_byTrial = dmd1Synapses[synapseIDX,:]
    glut_byTrial_noNans = glut_byTrial[~np.isnan(glut_byTrial)]
    lineareSynapseLearningCurve, _, _, _, _ =  stats.linregress(range(len(glut_byTrial_noNans)), glut_byTrial_noNans) #we only need the slope
    dmd1SynapseLearningCurves.append(lineareSynapseLearningCurve)

#sort this to find learning and antilearning synapses
dmd1SynapseLearningCurves = np.reshape(np.array(dmd1SynapseLearningCurves), (np.array(dmd1SynapseLearningCurves).shape[0],1))
dmd1_synapseLearning = np.where(dmd1SynapseLearningCurves>0)[0]         
dmd1_synapseAntiLearning = np.where(dmd1SynapseLearningCurves<0)[0]

####################################################################
#exact same thing but for trial 2
for trial in dmd2_byTrial:
    try:
        firstTrial = False
        glutTraces = np.array(trial['denoised'])
        synapticActivity_byTrial = np.empty(glutTraces.shape)
        nSynapses = glutTraces.shape[0]
        #the long way to remove bleaching for now... :(
        for synapseIDX in range(glutTraces.shape[0]):
            noNaNs_synapse = glutTraces[synapseIDX, ~np.isnan(glutTraces[synapseIDX,:])]
            m, b, _, _, _ = stats.linregress(range(len(noNaNs_synapse)), noNaNs_synapse)
            baseline_synapse = (m*np.arange(0, len(glutTraces[synapseIDX,:]), 1)) + b
            synapseActivity = glutTraces[synapseIDX, :] - baseline_synapse
            synapticActivity_byTrial[synapseIDX,:] = synapseActivity
            
        averagePerSynapse_byTrial = np.reshape(np.nanmean(synapticActivity_byTrial, axis=1), (nSynapses,1))
        dmd2Synapses.append(averagePerSynapse_byTrial)
        print(averagePerSynapse_byTrial.shape)
    
    except Exception:
        if firstTrial:
            firstTrial = False
            pass 
        nanSynapses = np.empty((nSynapses,1))
        nanSynapses[:] = np.nan
        dmd2Synapses.append(nanSynapses)
        print('NaN', nanSynapses.shape)

#glut averages by trial for each synapse    
dmd2Synapses = np.hstack(dmd2Synapses)

#find increase or decrease in activity
dmd2SynapseLearningCurves = []
for synapseIDX in range(dmd2Synapses.shape[0]):
    glut_byTrial = dmd2Synapses[synapseIDX,:]
    glut_byTrial_noNans = glut_byTrial[~np.isnan(glut_byTrial)]
    lineareSynapseLearningCurve, _, _, _, _ =  stats.linregress(range(len(glut_byTrial_noNans)), glut_byTrial_noNans) #we only need the slope
    dmd2SynapseLearningCurves.append(lineareSynapseLearningCurve)

#sort this to find learning and antilearning synapses
dmd2SynapseLearningCurves = np.reshape(np.array(dmd2SynapseLearningCurves), (np.array(dmd2SynapseLearningCurves).shape[0],1))
dmd2_synapseLearning = np.where(dmd2SynapseLearningCurves>0)[0]    
dmd2_synapseAntiLearning = np.where(dmd2SynapseLearningCurves<0)[0]

# %%
meanDMD1 = expmtSummary['exptSummary']['meanIM'][0][:,:,0]
meanDMD2 = expmtSummary['exptSummary']['meanIM'][1][:,:,0]
learning_color = (0, 255, 0)  # Green
antilearning_color = (0, 0, 255)  # Blue

#process it so cv2 can use it
meanDMD1_noNans = np.nan_to_num(meanDMD1, nan=0)
meanDMD1_normalized = cv2.normalize(meanDMD1_noNans, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
colorMeanIMG_dmd1 = cv2.cvtColor(meanDMD1_normalized, cv2.COLOR_GRAY2RGB)

# forDMD1 
plt.close('all')
fig3, ax3 = plt.subplots()
for footprintIDX in dmd1_synapseLearning:
    footPrintMask = dmd1_byTrial[1]['footprints'][:,:,footprintIDX]
    footprintIndices = np.where(footPrintMask>0, True, False)
    colorMeanIMG_dmd1[footprintIndices] = learning_color
    ax3.imshow(colorMeanIMG_dmd1)

for footprintIDX in dmd1_synapseAntiLearning:
    footPrintMask = dmd1_byTrial[1]['footprints'][:,:,footprintIDX]
    footprintIndices = np.where(footPrintMask>0, True, False)
    colorMeanIMG_dmd1[footprintIndices] = antilearning_color
    ax3.imshow(colorMeanIMG_dmd1)
ax3.tick_params(which = 'both', size = 0, labelsize = 0)    
# plt.show()
#########################################################################
#process it so cv2 can use it
meanDMD2_noNans = np.nan_to_num(meanDMD2, nan=0)
meanDMD2_normalized = cv2.normalize(meanDMD2_noNans, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
colorMeanIMG_dmd2 = cv2.cvtColor(meanDMD2_normalized, cv2.COLOR_GRAY2RGB)

# forDMD1 
# plt.close('all')
fig4, ax4 = plt.subplots()
for footprintIDX in dmd2_synapseLearning:
    footPrintMask = dmd2_byTrial[1]['footprints'][:,:,footprintIDX]
    footprintIndices = np.where(footPrintMask>0, True, False)
    colorMeanIMG_dmd2[footprintIndices] = learning_color
    ax4.imshow(colorMeanIMG_dmd1)

for footprintIDX in dmd2_synapseAntiLearning:
    footPrintMask = dmd2_byTrial[1]['footprints'][:,:,footprintIDX]
    footprintIndices = np.where(footPrintMask>0, True, False)
    colorMeanIMG_dmd2[footprintIndices] = antilearning_color
    ax4.imshow(colorMeanIMG_dmd2)
ax4.tick_params(which = 'both', size = 0, labelsize = 0)    
plt.show()


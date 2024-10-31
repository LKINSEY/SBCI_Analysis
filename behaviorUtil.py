#%%
import numpy as np
from scipy import stats
from mat73 import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import cv2
bks = matplotlib.rcsetup.interactive_bk
plt.switch_backend('WebAgg')
#%%
#reading harp bin files:
_SECONDS_PER_TICK = 32e-6
_payloadtypes = {
                1 : np.dtype(np.uint8),
                2 : np.dtype(np.uint16),
                4 : np.dtype(np.uint32),
                8 : np.dtype(np.uint64),
                129 : np.dtype(np.int8),
                130 : np.dtype(np.int16),
                132 : np.dtype(np.int32),
                136 : np.dtype(np.int64),
                68 : np.dtype(np.float32)
                }

def read_harp_bin(file):

    data = np.fromfile(file, dtype=np.uint8)

    if len(data) == 0:
        return None

    stride = data[1] + 2
    length = len(data) // stride
    payloadsize = stride - 12
    payloadtype = _payloadtypes[data[4] & ~0x10]
    elementsize = payloadtype.itemsize
    payloadshape = (length, payloadsize // elementsize)
    seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    ticks = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = ticks * _SECONDS_PER_TICK + seconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data, offset=11,
        strides=(stride, elementsize))

    if payload.shape[1] ==  1:
        ret_pd = pd.DataFrame(payload, index=seconds, columns= ["Value"])
        ret_pd.index.names = ['Seconds']

    else:
        ret_pd =  pd.DataFrame(payload, index=seconds)
        ret_pd.index.names = ['Seconds']

    return ret_pd

#assumes loadcell is highest precision, and aligns all timestamps recorded with behavior to match load cell time stamps
def alignToLoadCell(loadCellDF, toBeAlignedDF):
    #grab the time of each df
    baseTime = loadCellDF.index
    oldTime = toBeAlignedDF.index
    #find where each time from register fits in with time from loadcell register
    idxs = np.searchsorted(baseTime, oldTime)
    #make sure it works by clipping the ends
    idxs = np.clip(idxs, 1, max(idxs)-1)
    left = baseTime[idxs - 1]
    right = baseTime[idxs]
    #apply logic of determining if left or right indice is closer to replacement (drew on whiteboard at home)
    idxs_new = np.where(oldTime - left < right - oldTime, idxs-1, idxs)
    newTimes = baseTime[idxs_new]
    return newTimes

#aligns all data for a session and inputs it into a single format
def load_behavior_df(dataDir):
    '''
    NOTE: 
    you can ignore the caveats here, because we don't care about writing over a non-deep copy in this context...
    '''

    #Read Registers
    audioRegister               = read_harp_bin(dataDir + '/Behavior.harp/Register__PwmFrequencyDO1.bin' )
    loadCellRegister            = read_harp_bin(dataDir + '/LoadCells.harp/Register__LoadCellData.bin')
    motorData                   = pd.read_csv(dataDir + '/Operation/SpoutPosition.csv') #motor position
    digitalInputRegister        = read_harp_bin(dataDir + '/Behavior.harp/Register__DigitalInputState.bin') #licks and "waiting for acquisition"
    analogInputRegister         = read_harp_bin(dataDir + '/Behavior.harp/Register__AnalogData.bin') #BCI input
    microscopeTriggerRegister   = read_harp_bin(dataDir + '/Behavior.harp/Register__OutputSet.bin')
    
    #create a bigger "allData" df with loadcell data, and empty columns for other registers to eventually be filled in
    LCdata = pd.DataFrame(loadCellRegister[0])
    LCdata = LCdata.rename(columns={0:'LoadCell'})
    Tstamps = len(LCdata.values)
    fillInDF = (np.empty((Tstamps, 5)))
    fillInDF[:] = np.nan
    fillInDF = pd.DataFrame(fillInDF).set_index(LCdata.index)
    allData = pd.concat([LCdata, fillInDF], axis=1)
    allData = allData.rename(columns={ 0:'Audio', 1: 'Motor', 2:'Licks', 3:'ADC', 4:'HandShake'})

    #storing Audio
    audioAligned = alignToLoadCell(LCdata, audioRegister )
    audio = audioRegister.set_index(audioAligned)
    allData['Audio'][audioAligned] = np.array(audio.values).flatten()

    #storing motor
    motorTimes = motorData.pop('Seconds')
    motorAligned = alignToLoadCell(LCdata, motorData.set_index(motorTimes.values) )
    motor = motorData.set_index(motorAligned)
    allData['Motor'][  motor.index  ]= np.array(motor.values).flatten()
    allData['Motor'] = allData['Motor'].fillna(method='ffill')

    #storing lick values
    try:
        licksAligned = alignToLoadCell(LCdata, digitalInputRegister )
        licks = digitalInputRegister.set_index(licksAligned)
        # Treating licks differently because sometimes they don't happen
        allData['Licks'][licksAligned] = np.array(licks.values, dtype=np.int64).flatten()
    except Exception:
        pass

    #storing bci adc signal
    bciAligned = alignToLoadCell(LCdata, pd.DataFrame(analogInputRegister[0]))
    bci = analogInputRegister.set_index(bciAligned)#storing bci ADC
    try:
        bciInput = np.array(bci.values[:,0]).flatten()
        allData['ADC'][bciAligned]= bciInput[:]
    except Exception as e:
        print(e)

    #storing 2p handshake
    handshakeAligned = alignToLoadCell(LCdata,microscopeTriggerRegister)
    handshake = microscopeTriggerRegister.set_index(handshakeAligned)
    allData['HandShake'][handshakeAligned] = np.array(handshake.values).flatten()

    return allData




def trializeData(sessionData, viewTrials = False):
    startAudio = 500; rewardAudio = 100
    startPin = 0x1000; stopPin = 0x1
    trialIDX = 2

    trialDict = {}
    audioCues = sessionData['Audio'].dropna()
    goCues = audioCues[audioCues==startAudio].index[1:]  #for some reason the first go cue is incorrect, but thats ok we just do this
    rewardCues = audioCues[audioCues==rewardAudio].index

    triggers = sessionData['HandShake'].dropna()
    stopTriggers = triggers[((triggers.values).astype(np.int64)&stopPin)>0].index

    startTriggers = triggers[((triggers.values).astype(np.int64)&startPin)>0].index
    rewardIDX = 0
    for trialIDX in range(len(startTriggers)-1):
        data = {}
        data['Motor'] = sessionData['Motor'][np.logical_and(sessionData['Motor'].index>startTriggers[trialIDX] , sessionData['Motor'].index<stopTriggers[trialIDX+1])]
        data['LoadCell'] = sessionData['LoadCell'][np.logical_and(sessionData['LoadCell'].index>startTriggers[trialIDX] , sessionData['LoadCell'].index<stopTriggers[trialIDX+1])]
        data['ADC'] = sessionData['ADC'][np.logical_and(sessionData['ADC'].index>startTriggers[trialIDX] , sessionData['ADC'].index<stopTriggers[trialIDX+1])]
        if startTriggers[trialIDX] < rewardCues[rewardIDX] and stopTriggers[trialIDX+1] > rewardCues[rewardIDX]:
            data['Outcome'] = 1
            data['deltaT']  = rewardCues[rewardIDX]  - goCues[trialIDX]
            rewardIDX +=1
        else:
            data['Outcome'] = 0
            data['deltaT'] = 10 #max time between go cue and response allowed
        trialDict[trialIDX] = data

    if viewTrials:
        plt.close('all')
        for trialIDX in range(len(startTriggers)-1):
            thisTrial = sessionData['Motor'][np.logical_and(sessionData['Motor'].index>startTriggers[trialIDX] , sessionData['Motor'].index<stopTriggers[trialIDX+1])]
            print(startTriggers[trialIDX], '-->', stopTriggers[trialIDX])
            plt.figure()
            plt.plot(thisTrial.values)
        plt.show()
    return trialDict




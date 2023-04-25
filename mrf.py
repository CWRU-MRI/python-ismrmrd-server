import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import ctypes
import mrdhelper
from datetime import datetime

from tqdm import tqdm
from mrftools import *
import sys
from matplotlib import pyplot as plt
import scipy
from PIL import Image, ImageDraw, ImageFont

import os
import pickle
import hashlib

# Folder for debug output files
debugFolder = "/tmp/share/debug"
b1Folder = "/usr/share/b1-data"
dictionaryFolder = "/usr/share/dictionary-data"


# Configure dictionary simulation parameters
dictionaryName = "5pct"
percentStepSize=5; includeB1=False;  t1Range=(10,5000); t2Range=(1,500); b1Range=(0.5, 1.55); b1Stepsize=0.05; 
phaseRange=(-np.pi, np.pi); numSpins=10; numBatches=50
trajectoryFilepath="mrf_dependencies/trajectories/SpiralTraj_FOV250_256_uplimit1916_norm.bin"
densityFilepath="mrf_dependencies/trajectories/DCW_FOV250_256_uplimit1916.bin"

def ApplyXYZShift(svdData, header, acqHeaders, trajectories, matrixSizeOverride=None):
    shape = np.shape(svdData)
    numSVDComponents=shape[0]; numCoils=shape[1]; numPartitions=shape[2]; numReadoutPoints=shape[3]; numSpirals=shape[4]
    shiftedSvdData = torch.zeros_like(svdData)
    # For now, assume all spirals/partitions/etc have same offsets applied
    (x_shift, y_shift, z_shift) = CalculateVoxelOffsetAcquisitionSpace(header, acqHeaders[0,0,0], matrixSizeOverride=matrixSizeOverride)
    trajectories = torch.t(torch.tensor(np.array(trajectories)))
    x = torch.zeros((numPartitions, numReadoutPoints, numSpirals));
    y = torch.zeros((numPartitions, numReadoutPoints, numSpirals));
    partitions = torch.moveaxis(torch.arange(-0.5, 0.5, 1/numPartitions).expand((numReadoutPoints, numSpirals, numPartitions)), -1,0)
    trajectories = trajectories.expand((numPartitions, numReadoutPoints, numSpirals))
    x = torch.cos(-2*torch.pi*(x_shift*trajectories.real + y_shift*trajectories.imag + z_shift*partitions));
    y = torch.sin(-2*torch.pi*(x_shift*trajectories.real + y_shift*trajectories.imag + z_shift*partitions));
    print("K-Space x/y/z shift applied:", x_shift, y_shift, z_shift)
    return svdData*torch.complex(x,y)

def ApplyZShiftImageSpace(imageData, header, acqHeaders, matrixSizeOverride=None):
    (x_shift, y_shift, z_shift) = CalculateVoxelOffsetAcquisitionSpace(header, acqHeaders[0,0,0], matrixSizeOverride=matrixSizeOverride)
    return torch.roll(imageData, int(z_shift), dims=2)

def BatchPatternMatchViaMaxInnerProduct(signalTimecourses, dictionaryEntries, dictionaryEntryTimecourses, voxelsPerBatch=500, device=None):
    if(device==None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    signalsTransposed = torch.t(signalTimecourses)
    signalNorm = torch.linalg.norm(signalsTransposed, axis=1)[:,None]
    normalizedSignals = signalsTransposed / signalNorm

    simulationResults = torch.tensor(dictionaryEntryTimecourses, dtype=torch.complex64)
    simulationNorm = torch.linalg.norm(simulationResults, axis=0)
    normalizedSimulationResults = torch.t((simulationResults / simulationNorm)).to(device)

    numBatches = int(np.shape(normalizedSignals)[0]/voxelsPerBatch)
    patternMatches = np.empty((np.shape(normalizedSignals)[0]), dtype=DictionaryEntry)
    
    M0 = torch.zeros(np.shape(normalizedSignals)[0], dtype=torch.complex64)
    with tqdm(total=numBatches) as pbar:
        for i in range(numBatches):
            firstVoxel = i*voxelsPerBatch
            if i == (numBatches-1):
                lastVoxel = np.shape(normalizedSignals)[0]
            else:
                lastVoxel = firstVoxel+voxelsPerBatch
            batchSignals = normalizedSignals[firstVoxel:lastVoxel,:].to(device)
            innerProducts = torch.inner(batchSignals, normalizedSimulationResults)
            maxInnerProductIndices = torch.argmax(torch.abs(innerProducts), 1, keepdim=True)
            maxInnerProducts = torch.take_along_dim(innerProducts,maxInnerProductIndices,dim=1).squeeze()
            signalNorm_device = signalNorm[firstVoxel:lastVoxel].squeeze().to(device)
            simulationNorm_device = simulationNorm.to(device)[maxInnerProductIndices.squeeze().to(torch.long)]
            M0_device = signalNorm_device/simulationNorm_device
            M0[firstVoxel:lastVoxel] = M0_device.cpu()
            patternMatches[firstVoxel:lastVoxel] = dictionaryEntries[maxInnerProductIndices.squeeze().to(torch.long).cpu()].squeeze()
            pbar.update(1)
            del batchSignals, M0_device, signalNorm_device, simulationNorm_device
    del normalizedSimulationResults, dictionaryEntryTimecourses, dictionaryEntries, signalsTransposed, signalNorm, normalizedSignals, simulationResults
    del simulationNorm
    return patternMatches, M0


def AddText(image, text="NOT FOR DIAGNOSTIC USE", fontSize=12):
    matrixsize = np.shape(image)
    img = Image.fromarray(np.uint8(np.zeros((matrixsize[0:2]))))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("terminess.ttf", fontSize)
    _, _, w, h = draw.textbbox((0, 0), text, font=font)
    draw.text(((matrixsize[0]-w)/2, h/2), text, (255), font=font)
    overlay = np.array(img) > 0
    repeated = np.repeat(overlay[:,:,np.newaxis], matrixsize[2],axis=2)
    repeated = repeated + np.rot90(repeated)
    repeated = repeated + np.rot90(repeated)    
    repeated = repeated + np.rot90(repeated)    
    image[repeated] = np.max(image)
    return image

def LoadB1Map(matrixSize, resampleToMRFMatrixSize=True, deinterleave=True, performBinning=True):
    # Using header, generate a unique b1 filename. This is temporary
    try:
        b1Filename = "testB1"
        b1Data = np.load(b1Folder + "/" + b1Filename +".npy")
    except:
        print("No B1 map found")
        return np.array([])

    b1MapSize = np.array(np.shape(b1Data))
    print("B1 Input Size:", b1MapSize)
    if deinterleave:
        numSlices = b1MapSize[2]
        deinterleaved = np.zeros_like(b1Data)
        deinterleaved[:,:,np.arange(1,numSlices,2)] = b1Data[:,:,0:int(np.floor(numSlices/2))]
        deinterleaved[:,:,np.arange(0,numSlices-1,2)] = b1Data[:,:,int(np.floor(numSlices/2)):numSlices]
        b1Data = deinterleaved
    if resampleToMRFMatrixSize:
        b1Data = scipy.ndimage.zoom(b1Data, matrixSize/b1MapSize, order=5)
        b1Data = np.flip(b1Data, axis=2)
        b1Data = np.rot90(b1Data, axes=(0,1))
        b1Data = np.flip(b1Data, axis=0)
    print("B1 Output Size:", np.shape(b1Data))
    return b1Data
        
def performB1Binning(b1Data, b1Range, b1Stepsize, b1IdentityValue=800):
    b1Bins = np.arange(b1Range[0], b1Range[1], b1Stepsize)
    b1Clipped = np.clip(b1Data, np.min(b1Bins)*b1IdentityValue, np.max(b1Bins)*b1IdentityValue)
    b1Binned = b1Bins[np.digitize(b1Clipped, b1Bins*b1IdentityValue, right=True)]
    print("Binned B1 Shape: ", np.shape(b1Binned))
    return b1Binned

def GenerateDictionaryLookupTables(dictionary):
    uniqueT1s = np.unique(dictionary.entries['T1'])
    uniqueT2s = np.unique(dictionary.entries['T2'])

    dictionary2DIndexLookupTable = []
    dictionaryEntries2D = np.zeros((len(uniqueT1s), len(uniqueT2s)), dtype=DictionaryEntry)
    dictionary1DIndexLookupTable = np.zeros((len(uniqueT1s), len(uniqueT2s)), dtype=int)
    #printAllTensors()
    for dictionaryIndex in tqdm(range(len(dictionary.entries))):
        entry = dictionary.entries[dictionaryIndex]
        T1index = np.where(uniqueT1s == entry['T1'])[0]
        T2index = np.where(uniqueT2s == entry['T2'])[0]
        dictionaryEntries2D[T1index, T2index] = entry
        dictionary1DIndexLookupTable[T1index, T2index] = dictionaryIndex
        dictionary2DIndexLookupTable.append([T1index,T2index])
    dictionary2DIndexLookupTable = np.array(dictionary2DIndexLookupTable)
    return dictionary1DIndexLookupTable, dictionary2DIndexLookupTable

def PatternMatchingViaMaxInnerProduct(combined, dictionary, simulation, voxelsPerBatch=500, b1Binned=None, device=None,):
    if(device==None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    dictionary1DIndexLookupTable, dictionary2DIndexLookupTable = GenerateDictionaryLookupTables(dictionary)
    sizes = np.shape(combined)
    numSVDComponents=sizes[0]; matrixSize=sizes[1:4]
    patternMatches = np.empty((matrixSize), dtype=DictionaryEntry)
    M0 = torch.zeros((matrixSize), dtype=torch.complex64)
    if b1Binned is not None:
        for uniqueB1 in np.unique(b1Binned):
            print(uniqueB1)
            if uniqueB1 == 0:
                patternMatches[b1Binned==uniqueB1] = 0
            else:
                signalTimecourses = combined[:,b1Binned == uniqueB1]
                simulationTimecourses = torch.t(torch.t(torch.tensor(simulation.truncatedResults))[(np.argwhere(dictionary.entries['B1'] == uniqueB1))].squeeze())
                dictionaryEntries = dictionary.entries[(np.argwhere(dictionary.entries['B1'] == uniqueB1))]
                signalTimecourses = combined[:,b1Binned == uniqueB1]
                patternMatches[b1Binned == uniqueB1], M0[b1Binned == uniqueB1] = BatchPatternMatchViaMaxInnerProduct(signalTimecourses,dictionaryEntries,simulationTimecourses,voxelsPerBatch, device=device)
    else:
        signalTimecourses = torch.reshape(combined, (numSVDComponents,-1))
        if(dictionary.entries['B1'][0]):
            simulationTimecourses = torch.t(torch.t(torch.tensor(simulation.truncatedResults))[(np.argwhere(dictionary.entries['B1'] == 1))].squeeze())
            dictionaryEntries = dictionary.entries[(np.argwhere(dictionary.entries['B1'] == 1))]
        else:   
            simulationTimecourses = torch.tensor(simulation.truncatedResults)
            dictionaryEntries = dictionary.entries
        patternMatches, M0 = BatchPatternMatchViaMaxInnerProduct(signalTimecourses, dictionaryEntries, simulationTimecourses, voxelsPerBatch, device=device)
    patternMatches = np.reshape(patternMatches, (matrixSize))
    M0 = np.reshape(M0, (matrixSize)).numpy()
    M0 = np.nan_to_num(M0)
    return patternMatches, M0

# Takes data input as: [cha z y x], [z y x], or [y x]
def PopulateISMRMRDImage(header, data, acquisition, image_index, colormap=None, window=None, level=None, comment=""):
    image = ismrmrd.Image.from_array(data.transpose(), acquisition=acquisition, transpose=False)
    image.image_index = image_index

    # Set field of view
    image.field_of_view = (ctypes.c_float(header.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(header.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(header.encoding[0].reconSpace.fieldOfView_mm.z))

    if colormap is None:
        colormap = ""
    if window is None:
        window = np.max(data)
    if level is None:
        level = np.max(data)/2

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           str(level),
                         'WindowWidth':            str(window), 
                         'GADGETRON_ColorMap':     colormap,
                         'GADGETRON_ImageComment': comment})

    # Add image orientation directions to MetaAttributes if not already present
    if meta.get('ImageRowDir') is None:
        meta['ImageRowDir'] = ["{:.18f}".format(image.getHead().read_dir[0]), "{:.18f}".format(image.getHead().read_dir[1]), "{:.18f}".format(image.getHead().read_dir[2])]

    if meta.get('ImageColumnDir') is None:
        meta['ImageColumnDir'] = ["{:.18f}".format(image.getHead().phase_dir[0]), "{:.18f}".format(image.getHead().phase_dir[1]), "{:.18f}".format(image.getHead().phase_dir[2])]

    xml = meta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has %d elements", image.data.size)

    image.attribute_string = xml
    return image
            
def process(connection, config, metadata):
    logging.info("Config: \n%s", config)
    logging.info("Metadata: \n%s", metadata)
    
    header = metadata
    enc = metadata.encoding[0]
    numSpirals = enc.encodingLimits.kspace_encoding_step_1.maximum+1; print("Spirals:", numSpirals)
    numMeasuredPartitions = enc.encodingLimits.kspace_encoding_step_2.maximum+1; print("Measured Partitions:", numMeasuredPartitions)
    centerMeasuredPartition = enc.encodingLimits.kspace_encoding_step_2.center; print("Center Measured Partition:", centerMeasuredPartition) # Fix this in stylesheet
    numSets = enc.encodingLimits.set.maximum+1; print("Sets:", numSets)
    numCoils = header.acquisitionSystemInformation.receiverChannels; print("Coils:", numCoils)
    matrixSize = np.array([enc.reconSpace.matrixSize.x,enc.reconSpace.matrixSize.y,enc.reconSpace.matrixSize.z]); print("Matrix Size:", matrixSize)
    numUndersampledPartitions = matrixSize[2]; print("Undersampled Partitions:", numUndersampledPartitions)
    undersamplingRatio = 1
    if(numUndersampledPartitions > 1): # Hack, may not work for multislice 2d
        undersamplingRatio = int(numUndersampledPartitions / (centerMeasuredPartition * 2)); print("Undersampling Ratio:", undersamplingRatio)
    usePartialFourier = False
    if(numMeasuredPartitions*undersamplingRatio < numUndersampledPartitions):
        usePartialFourier = True
        partialFourierRatio = numMeasuredPartitions / (numUndersampledPartitions/undersamplingRatio)
        print("Measured partitions is less than expected for undersampling ratio - assuming partial fourier acquisition with ratio:", partialFourierRatio)
    
    # Set up sequence parameter arrays
    numTimepoints = numSets*numSpirals
    TRs = np.zeros((numTimepoints, numMeasuredPartitions))
    TEs = np.zeros((numTimepoints, numMeasuredPartitions))
    FAs = np.zeros((numTimepoints, numMeasuredPartitions))
    PHs = np.zeros((numTimepoints, numMeasuredPartitions))
    IDs = np.zeros((numTimepoints, numMeasuredPartitions))
    
    # Set up raw data and header arrays
    rawdata = None
    #pilotToneData = np.zeros([numCoils, numUndersampledPartitions, numSpirals, numSets], dtype=np.complex64)
    acqHeaders = np.empty((numUndersampledPartitions, numSpirals, numSets), dtype=ismrmrd.Acquisition)
    discardPre=0;discardPost=0
    
    # Process data as it comes in
    try:
        for acq in connection:
            if acq is None:
                break
            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) or acq.isFlagSet(ismrmrd.ACQ_IS_PHASECORR_DATA):
                continue
            else:
                acqHeader = acq.getHead()
                measuredPartition = acq.idx.kspace_encode_step_2
                undersampledPartition = acq.user_int[1]
                spiral = acq.idx.kspace_encode_step_1
                set = acq.idx.set
                timepoint = acq.user_int[0]
                TRs[timepoint, measuredPartition] = acq.user_int[2]            
                TEs[timepoint, measuredPartition] = acq.user_int[3]
                FAs[timepoint, measuredPartition] = acq.user_int[5] # Use actual FA not requested
                PHs[timepoint, measuredPartition] = acq.user_int[6] 
                IDs[timepoint, measuredPartition] = spiral
                #print(measuredPartition, undersampledPartition, spiral, set)
                acqHeaders[undersampledPartition, spiral, set] = acqHeader
                if rawdata is None:
                    discardPre = int(acqHeader.discard_pre / 2); print("Discard Pre:", discardPre) # Fix doubling in sequence - weird;
                    discardPost = discardPre + 1916; print("Discard Post:", discardPost) # Fix in sequence
                    numReadoutPoints = discardPost-discardPre; print("Readout Points:", numReadoutPoints)
                    rawdata = np.zeros([numCoils, numUndersampledPartitions, numReadoutPoints, numSpirals, numSets], dtype=np.complex64)
                readout = acq.data[:, discardPre:discardPost]
                #readout_fft = torch.fft.fft(torch.tensor(readout), dim=1)
                #(pt_point,_) = torch.max(torch.abs(readout_fft), dim=1)
                #pilotToneData[:, undersampledPartition, spiral, set] = pt_point.numpy() # Change from max to max of
                rawdata[:, undersampledPartition, :, spiral, set] = readout
    except Exception as e:
        logging.exception(e)   
        connection.send_close()     

    B1map = LoadB1Map(matrixSize)
    if(np.size(B1map)!=0):
        B1map_binned = performB1Binning(B1map, b1Range, b1Stepsize, b1IdentityValue=800)
        dictionary = DictionaryParameters.GenerateFixedPercent(dictionaryName, percentStepSize=percentStepSize, t1Range=t1Range, t2Range=t2Range, includeB1=True, b1Range=b1Range, b1Stepsize=b1Stepsize)
    else:
        B1map_binned = None
        dictionary = DictionaryParameters.GenerateFixedPercent(dictionaryName, percentStepSize=percentStepSize, t1Range=t1Range, t2Range=t2Range, includeB1=False, b1Range=None, b1Stepsize=None)

    ## Simulate the Dictionary
    sequence = SequenceParameters("on-the-fly", SequenceType.FISP)
    sequence.Initialize(TRs[:,0]/(1000*1000), TEs[:,0]/(1000*1000), FAs[:,0]/(100), PHs[:,0]/(100), IDs[:,0])
    sequenceHash = hashlib.sha256(pickle.dumps(sequence)).hexdigest()
    dictionaryPath = dictionaryFolder+ "/" +sequenceHash+".sequence"
    print(dictionaryPath)

    ## Check if dictionary already exists
    if (os.path.isfile(dictionaryPath)):
        print("Dictionary already exists. Using local copy.")
        filehandler = open(dictionaryPath,'rb')
        simulation = pickle.load(filehandler) 
        filehandler.close()

    else:        
        print("Dictionary not found. Simulating. ")
        simulation = Simulation(sequence, dictionary, phaseRange=phaseRange, numSpins=numSpins)
        simulation.Execute(numBatches=numBatches)
        simulation.CalculateSVD(desiredSVDPower=0.98)
        print("Simulated", numSpirals*numSets, "timepoints")
        del simulation.results
        filehandler = open(dictionaryPath, 'wb')
        pickle.dump(simulation, filehandler)
        filehandler.close()

    ## Run the Reconstruction
    svdData = ApplySVDCompression(rawdata, simulation, device=torch.device("cpu"))
    (trajectoryBuffer,trajectories,densityBuffer,_) = LoadSpirals(trajectoryFilepath, densityFilepath, numSpirals)
    svdData = ApplyXYZShift(svdData, header, acqHeaders, trajectories)
    nufftResults = PerformNUFFTs(svdData, trajectoryBuffer, densityBuffer, matrixSize, matrixSize*2)
    del svdData
    coilImageData = PerformThroughplaneFFT(nufftResults)
    del nufftResults
    imageData, coilmaps = PerformWalshCoilCombination(coilImageData)
    imageMask = GenerateMaskFromCoilmaps(coilmaps)

    print("Starting normal pattern match")
    patternMatchResults, M0 = PatternMatchingViaMaxInnerProduct(imageData*imageMask, dictionary, simulation, b1Binned = B1map_binned)
    print("Finished normal pattern match")

    # Generate Classification Maps from Timecourses and Known Tissue Timecourses

######################################
    print("Starting WM/GM/CSF Seperation")
    ## Run for all pixels
    maskedImages = imageData*imageMask
    shape = np.shape(maskedImages)
    timecourses = maskedImages.reshape(shape[0], -1)

    ## Set up coefficient dictionary
    coefficientDictionaryEntries = []
    stepSize = 1/(10**2)
    roundingFactor = 1/stepSize
    maxSum = 1

    for aValue in np.arange(0, maxSum, stepSize):
        remainingForB = maxSum - aValue
        if(remainingForB < stepSize):
            coefficientDictionaryEntries.append([aValue, remainingForB, 0])
        else:   
            for bValue in np.arange(0,remainingForB, stepSize):
                remainingForC = remainingForB - bValue
                coefficientDictionaryEntries.append([aValue, bValue, remainingForC])
    coefficientDictionaryEntries = np.array(coefficientDictionaryEntries)
    coefficientDictionaryEntries = np.round(coefficientDictionaryEntries*roundingFactor)/roundingFactor
    sums = np.sum(coefficientDictionaryEntries, axis=1)
    coefficientDictionaryEntries = np.array([tuple(i) for i in coefficientDictionaryEntries], dtype=DictionaryEntry)

    ## Timecourse Equation for a voxel
    ## Gm(t) = sum across dictionary entries of e^-1*(((T1_gm-T1)/sigmaT1_gm)**2 + ((T2_gm-T2)/sigmaT2_gm)**2)
    T1_wm = WHITE_MATTER_3T[0]['T1']; sigmaT1_wm = 0.01
    T2_wm = WHITE_MATTER_3T[0]['T2']; sigmaT2_wm = 0.01
    T1_gm = GREY_MATTER_3T[0]['T1']; sigmaT1_gm = 0.01
    T2_gm = GREY_MATTER_3T[0]['T2']; sigmaT2_gm = 0.01
    T1_csf = CSF_3T[0]['T1']; sigmaT1_csf = 0.01
    T2_csf = CSF_3T[0]['T2']; sigmaT2_csf = 0.01
    T1 = dictionary.entries['T1']; T2 = dictionary.entries['T2']
    WmWeights = np.exp( -1 * ( ((T1_wm - T1)/sigmaT1_wm )**2 + ( (T2_wm-T2)/sigmaT2_wm )**2 ))
    GmWeights = np.exp( -1 * ( ((T1_gm - T1)/sigmaT1_gm )**2 + ( (T2_gm-T2)/sigmaT2_gm )**2 )) 
    CsfWeights = np.exp( -1 * ( ((T1_csf - T1)/sigmaT1_csf )**2 + ( (T2_csf-T2)/sigmaT2_csf )**2 )) 

    ## Create timecourses for WM/GM/CSF based on the above 
    WM = np.sum(simulation.truncatedResults * WmWeights, axis=1); GM = np.sum(simulation.truncatedResults * GmWeights, axis=1); CSF = np.sum(simulation.truncatedResults * CsfWeights, axis=1)
    coefficientDictionaryTimecourses = []
    for coefficients in coefficientDictionaryEntries:
        coefficientTimecourse = coefficients['T1'] * WM + coefficients['T2'] * GM + coefficients['B1'] * CSF
        coefficientDictionaryTimecourses.append(coefficientTimecourse)
    coefficientDictionaryTimecourses = np.array(coefficientDictionaryTimecourses).transpose()

    # Perform Coefficient-Space Pattern Matching
    coefficientPatternMatches,coefficientM0 = BatchedPatternMatchViaMaxInnerProduct(timecourses.to(torch.cfloat), coefficientDictionaryEntries, torch.tensor(coefficientDictionaryTimecourses).to(torch.cfloat))
    normalizedCoefficients = coefficientPatternMatches.view((np.dtype('<f4'), len(coefficientPatternMatches.dtype.names)))

    wmFractionMap = np.reshape(normalizedCoefficients[:,0], (matrixSize)) * 10000
    gmFractionMap = np.reshape(normalizedCoefficients[:,1], (matrixSize)) * 10000
    csfFractionMap = np.reshape(normalizedCoefficients[:,2], (matrixSize)) * 10000

######################################
    print("emitting images")
    images = []
    # Data is ordered [x y z] in patternMatchResults and M0. Need to reorder to [z y x] for ISMRMRD, and split T1/T2 apart
    T1map = patternMatchResults['T1'] * 1000 # to milliseconds
    T1image = PopulateISMRMRDImage(header, AddText(T1map), acqHeaders[0,0,0], 0, window=2500, level=1250, comment="T1_ms")
    images.append(T1image)  

    T2map = patternMatchResults['T2'] * 1000 # to milliseconds
    T2image = PopulateISMRMRDImage(header, AddText(T2map), acqHeaders[0,0,0], 2, window=500, level=200, comment="T2_ms")
    images.append(T2image)   

    M0map = (np.abs(M0) / np.max(np.abs(M0))) * 2**12
    M0image = PopulateISMRMRDImage(header, AddText(M0map), acqHeaders[0,0,0], 3, comment="M0")
    images.append(M0image)   

    if(np.size(B1map) != 0):
        B1image = PopulateISMRMRDImage(header, AddText(B1map), acqHeaders[0,0,0], 5, comment="B1")
        images.append(B1image)  

    WMimage = PopulateISMRMRDImage(header, AddText(wmFractionMap), acqHeaders[0,0,0], 5, comment="WM_Fraction")
    images.append(WMimage)   

    GMimage = PopulateISMRMRDImage(header, AddText(gmFractionMap), acqHeaders[0,0,0], 6, comment="GM_Fraction")
    images.append(GMimage)   

    CSFimage = PopulateISMRMRDImage(header, AddText(csfFractionMap), acqHeaders[0,0,0], 7, comment="CSF_Fraction")
    images.append(CSFimage)  
    try:
        print("Mask Image:", np.shape(imageMask))
        MASKimage = PopulateISMRMRDImage(header, AddText(imageMask*1000), acqHeaders[0,0,0],8,comment="mask")
        images.append(MASKimage)
    except:
        print("Couldn't send Mask Image")
    connection.send_image(images)
    connection.send_close()

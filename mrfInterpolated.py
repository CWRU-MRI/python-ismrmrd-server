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

# Folder for debug output files
debugFolder = "/tmp/share/debug"
b1Folder = "/usr/share/b1-data"

# Configure dictionary simulation parameters
dictionaryName = "5pct"
percentStepSize=5; includeB1=False;  t1Range=(1,5000); t2Range=(1,500); b1Range=(0.5, 1.55); b1Stepsize=0.10; 
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

def BatchPatternMatchViaMaxInnerProductWithInterpolation(signalTimecourses, dictionaryEntries, dictionaryEntryTimecourses, dictionary1DIndexLookupTable, dictionary2DIndexLookupTable, voxelsPerBatch=500, device=None, radius=1):
    if(device==None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    with torch.no_grad():
        signalsTransposed = torch.t(signalTimecourses)
        signalNorm = torch.linalg.norm(signalsTransposed, axis=1)[:,None]
        normalizedSignals = signalsTransposed / signalNorm

        simulationResults = torch.tensor(dictionaryEntryTimecourses, dtype=torch.complex64)
        simulationNorm = torch.linalg.norm(simulationResults, axis=0)
        normalizedSimulationResults = torch.t((simulationResults / simulationNorm)).to(device)

        numBatches = int(np.shape(normalizedSignals)[0]/voxelsPerBatch)
        patternMatches = np.empty((np.shape(normalizedSignals)[0]), dtype=DictionaryEntry)
        interpolatedMatches = np.empty((np.shape(normalizedSignals)[0]), dtype=DictionaryEntry)

        offsets = np.mgrid[-1*radius:radius+1, -1*radius:radius+1]
        numNeighbors = np.shape(offsets)[1]*np.shape(offsets)[2]
        
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

                indices = dictionary2DIndexLookupTable[maxInnerProductIndices.squeeze().to(torch.long).cpu()].squeeze()

                numVoxels = len(maxInnerProductIndices)
                neighbor2DIndices = np.reshape(indices.repeat(numNeighbors,axis=1),(np.shape(indices)[0], np.shape(indices)[1],np.shape(offsets)[1], np.shape(offsets)[2])) + offsets
                neighbor2DIndices[:,0,:,:] = np.clip(neighbor2DIndices[:,0,:,:], 0, np.shape(dictionary1DIndexLookupTable)[0]-1)
                neighbor2DIndices[:,1,:,:] = np.clip(neighbor2DIndices[:,1,:,:], 0, np.shape(dictionary1DIndexLookupTable)[1]-1)

                neighborDictionaryIndices = torch.tensor(dictionary1DIndexLookupTable[neighbor2DIndices[:,0,:,:], neighbor2DIndices[:,1,:,:]].reshape(numVoxels, -1))
                neighborDictionaryEntries = dictionaryEntries[neighborDictionaryIndices]
                neighborInnerProducts = torch.take_along_dim(innerProducts.cpu(),neighborDictionaryIndices,dim=1).squeeze()
                
                ## Perform pseudoinverse solve for dictionary-space parabolic function coefficients
                innerProducts = torch.abs(neighborInnerProducts).to(torch.float32).unsqueeze(1)
                T1s = torch.tensor(neighborDictionaryEntries['T1']); T2s = torch.tensor(neighborDictionaryEntries['T2'])
                basis = torch.stack([T1s**2, T2s**2, T1s*T2s, T1s, T2s, torch.ones(np.shape(neighborDictionaryEntries))])
                basis = torch.moveaxis(basis,1,0).squeeze()
                pinv = torch.linalg.pinv(basis)
                coefficients = torch.bmm(innerProducts, pinv).squeeze()

                ## Perform inversion solve for partial derivatives of 0 of dictionary-space parabolic function
                A = torch.zeros((np.shape(coefficients)[0], 2, 2))
                f_T1T2 = torch.zeros((np.shape(coefficients)[0], 2))
                A[:,0,0] = 2*coefficients[:,0]; A[:,0,1] = coefficients[:,2];  A[:,1,0] = coefficients[:,2];  A[:,1,1] =  2*coefficients[:,1]
                f_T1T2[:,0] = -1*coefficients[:,3]; f_T1T2[:,1] = -1*coefficients[:,4]
                invA = torch.linalg.pinv(A)
                minimumT1T2 = torch.bmm(f_T1T2.unsqueeze(1), invA).squeeze().numpy()

                interpolatedValues = np.zeros((numVoxels),dtype=DictionaryEntry)
                for voxel in range(np.shape(minimumT1T2)[0]):
                    interpolatedValues[voxel]['T1'] = minimumT1T2[voxel][0]
                    interpolatedValues[voxel]['T2'] = minimumT1T2[voxel][1]
                interpolatedMatches[firstVoxel:lastVoxel] = interpolatedValues
                pbar.update(1)
                del batchSignals, M0_device, signalNorm_device, simulationNorm_device

        del normalizedSimulationResults, dictionaryEntryTimecourses, dictionaryEntries, signalsTransposed, signalNorm, normalizedSignals, simulationResults
        del simulationNorm
        return patternMatches,interpolatedMatches, M0


def LoadB1Map(matrixSize, resampleToMRFMatrixSize=True, deinterleave=True, performBinning=True):
    # Using header, generate a unique b1 filename. This is temporary
    try:
        b1Filename = "testB1"
        b1Data = np.load(b1Folder + "/" + b1Filename +".npy")
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
        print("B1 Output Size:", np.shape(b1Data))
        return b1Data
    except:
        print("No B1 map found")
        return None
    
def performB1Binning(b1Data, b1Range, b1Stepsize, b1IdentityValue=800):
    b1Bins = np.arange(b1Range[0], b1Range[1], b1Stepsize)
    b1Clipped = np.clip(b1Data, np.min(b1Bins)*b1IdentityValue, np.max(b1Bins)*b1IdentityValue)
    b1Binned = b1Bins[np.digitize(b1Clipped, b1Bins*b1IdentityValue, right=True)]
    print("Binned B1 Shape: ", b1Binned)
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

def PatternMatchingViaMaxInnerProductWithInterpolation(combined, dictionary, simulation, voxelsPerBatch=500, b1Binned=None, device=None,):
    if(device==None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    dictionary1DIndexLookupTable, dictionary2DIndexLookupTable = GenerateDictionaryLookupTables(dictionary)
    sizes = np.shape(combined)
    numSVDComponents=sizes[0]; matrixSize=sizes[1:4]
    patternMatches = np.empty((matrixSize), dtype=DictionaryEntry)
    interpolatedMatches = np.empty((matrixSize), dtype=DictionaryEntry)
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
                patternMatches[b1Binned == uniqueB1], interpolatedMatches[b1Binned == uniqueB1], M0[b1Binned == uniqueB1] = BatchPatternMatchViaMaxInnerProductWithInterpolation(signalTimecourses,dictionaryEntries,simulationTimecourses, dictionary1DIndexLookupTable, dictionary2DIndexLookupTable, voxelsPerBatch=voxelsPerBatch, device=device)
    else:
        signalTimecourses = torch.reshape(combined, (numSVDComponents,-1))
        if(dictionary.entries['B1'][0]):
            simulationTimecourses = torch.t(torch.t(torch.tensor(simulation.truncatedResults))[(np.argwhere(dictionary.entries['B1'] == 1))].squeeze())
            dictionaryEntries = dictionary.entries[(np.argwhere(dictionary.entries['B1'] == 1))]
        else:   
            simulationTimecourses = torch.tensor(simulation.truncatedResults)
            dictionaryEntries = dictionary.entries
        patternMatches, interpolatedMatches, M0 = BatchPatternMatchViaMaxInnerProductWithInterpolation(signalTimecourses, dictionaryEntries, simulationTimecourses, dictionary1DIndexLookupTable, dictionary2DIndexLookupTable, voxelsPerBatch=voxelsPerBatch, device=device)
    patternMatches = np.reshape(patternMatches, (matrixSize))
    interpolatedMatches = np.reshape(interpolatedMatches, (matrixSize))
    M0 = np.reshape(M0, (matrixSize)).numpy()
    M0 = np.nan_to_num(M0)
    return patternMatches, interpolatedMatches, M0


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
    if(B1map != None):
        B1map_binned = performB1Binning(B1map, b1Range, b1Stepsize, b1IdentityValue=800)
        dictionary = DictionaryParameters.GenerateFixedPercent(dictionaryName, percentStepSize=percentStepSize, includeB1=includeB1, t1Range=t1Range, t2Range=t2Range, includeB1=True, b1Range=b1Range, b1Stepsize=b1Stepsize)
    else:
        B1map_binned = None
        dictionary = DictionaryParameters.GenerateFixedPercent(dictionaryName, percentStepSize=percentStepSize, includeB1=includeB1, t1Range=t1Range, t2Range=t2Range, includeB1=False, b1Range=None, b1Stepsize=None)

    ## Simulate the Dictionary
    sequence = SequenceParameters("on-the-fly", SequenceType.FISP)
    sequence.Initialize(TRs[:,0]/(1000*1000), TEs[:,0]/(1000*1000), FAs[:,0]/(100), PHs[:,0]/(100), IDs[:,0])
    dictionary = DictionaryParameters.GenerateFixedPercent(dictionaryName, percentStepSize=percentStepSize, includeB1=includeB1, t1Range=t1Range, t2Range=t2Range, b1Range=b1Range, b1Stepsize=b1Stepsize)
    simulation = Simulation(sequence, dictionary, phaseRange=phaseRange, numSpins=numSpins)
    simulation.Execute(numBatches=numBatches)
    simulation.CalculateSVD(desiredSVDPower=0.995)
    print("Simulated", numSpirals*numSets, "timepoints")
    del simulation.results

    ## Setup for parabolic interpolation
    uniqueT1s = np.unique(dictionary.entries['T1'])
    uniqueT2s = np.unique(dictionary.entries['T2'])

    dictionary2DIndexLookupTable = []
    dictionaryEntries2D = np.zeros((len(uniqueT1s), len(uniqueT2s)), dtype=DictionaryEntry)
    dictionary1DIndexLookupTable = np.zeros((len(uniqueT1s), len(uniqueT2s)), dtype=int)

    for dictionaryIndex in tqdm(range(len(dictionary.entries))):
        entry = dictionary.entries[dictionaryIndex]
        T1index = np.where(uniqueT1s == entry['T1'])[0]
        T2index = np.where(uniqueT2s == entry['T2'])[0]
        dictionaryEntries2D[T1index, T2index] = entry
        dictionary1DIndexLookupTable[T1index, T2index] = dictionaryIndex
        dictionary2DIndexLookupTable.append([T1index,T2index]);
    dictionary2DIndexLookupTable = np.array(dictionary2DIndexLookupTable)

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
    print("Starting interpolated pattern match")
    patternMatchResults, interpolatedResults, M0 = PatternMatchingViaMaxInnerProductWithInterpolation(imageData*imageMask, dictionary, simulation, b1Binned = B1map_binned)
    print("Finished interpolated pattern match")

    # Generate Classification Maps from Timecourses and Known Tissue Timecourses

######################################

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

    images = []
    # Data is ordered [x y z] in patternMatchResults and M0. Need to reorder to [z y x] for ISMRMRD, and split T1/T2 apart
    T1map = patternMatchResults['T1'] * 1000 # to milliseconds
    #T1image = PopulateISMRMRDImage(header, T1map, acqHeaders[0,0,0], 0, colormap="FingerprintingT1.pal", window=2500, level=1250)
    T1image = PopulateISMRMRDImage(header, T1map, acqHeaders[0,0,0], 0, window=2500, level=1250, comment="T1_ms")
    images.append(T1image)   

    T1mapInterpolated = interpolatedResults['T1'] * 1000 # to milliseconds
    T1imageInterpolated = PopulateISMRMRDImage(header, T1mapInterpolated, acqHeaders[0,0,0], 1, window=2500, level=1250, comment="T1_ms_interpolated")
    images.append(T1imageInterpolated)   

    T2map = patternMatchResults['T2'] * 1000 # to milliseconds
    #T2image = PopulateISMRMRDImage(header, T2map, acqHeaders[0,0,0], 1, colormap="FingerprintingT2.pal", window=500, level=200)   
    T2image = PopulateISMRMRDImage(header, T2map, acqHeaders[0,0,0], 2, window=500, level=200, comment="T2_ms")
    images.append(T2image)   
    
    T2mapInterpolated = interpolatedResults['T2'] * 1000 # to milliseconds
    T2imageInterpolated = PopulateISMRMRDImage(header, T2mapInterpolated, acqHeaders[0,0,0], 3, window=500, level=200, comment="T2_ms_interpolated")
    images.append(T2imageInterpolated) 

    M0map = (np.abs(M0) / np.max(np.abs(M0))) * 2**12
    M0image = PopulateISMRMRDImage(header, M0map, acqHeaders[0,0,0], 4, comment="M0")
    images.append(M0image)   

    if(B1map):
        B1image = PopulateISMRMRDImage(header, B1map, acqHeaders[0,0,0], 5, comment="B1")
        images.append(B1image)  

    WMimage = PopulateISMRMRDImage(header, wmFractionMap, acqHeaders[0,0,0], 6, comment="WM_Fraction")
    images.append(WMimage)   

    GMimage = PopulateISMRMRDImage(header, gmFractionMap, acqHeaders[0,0,0], 7, comment="GM_Fraction")
    images.append(GMimage)   

    CSFimage = PopulateISMRMRDImage(header, csfFractionMap, acqHeaders[0,0,0], 8, comment="CSF_Fraction")
    images.append(CSFimage)   

    connection.send_image(images)
    connection.send_close()

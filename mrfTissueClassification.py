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
from scipy import stats

# Folder for debug output files
debugFolder = "/tmp/share/debug"

# Configure dictionary simulation parameters
dictionaryName = "5pctNoB1"
fixedStepSize=5; includeB1=False;  t1Range=(5,5000); t2Range=(5,500); b1Range=None; b1Stepsize=None; #b1Range=(0.1, 2.5); b1Stepsize=0.1; #t1Range=(50,5000); t2Range=(1,500)
phaseRange=(-np.pi, np.pi); numSpins=50; numBatches=20
trajectoryFilepath="mrf_dependencies/trajectories/SpiralTraj_FOV250_256_uplimit1916_norm.bin"
densityFilepath="mrf_dependencies/trajectories/DCW_FOV250_256_uplimit1916.bin"
     
# Takes data input as: [cha z y x], [z y x], or [y x]
def PopulateISMRMRDImage(header, data, acquisition, image_index, colormap=None, window=None, level=None):
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
                         'GADGETRON_ColorMap':     colormap     })

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

def CoilConsensusPatternMatching(coilImageData, dictionary, simulation, dictionaryStepSizePct=0.03):
    sizes = np.shape(coilImageData)
    numSVDComponents=sizes[0]; numCoils=sizes[1]; matrixSize=sizes[2:5]
    patternMatchResults = np.empty((sizes[1:5]), dtype=DictionaryEntry)    
    M0 = np.zeros((sizes[1:5]))
    means = np.zeros((sizes[2],sizes[3],sizes[4]), dtype=DictionaryEntry)
    stddevs = np.zeros((sizes[2],sizes[3],sizes[4]), dtype=DictionaryEntry)
    modes = np.zeros((sizes[2],sizes[3],sizes[4]), dtype=DictionaryEntry)
    confidence = np.zeros((sizes[2],sizes[3],sizes[4]), dtype=DictionaryEntry)

    for coil in np.arange(0,numCoils):
        patternMatchResults[coil,:,:,:], M0[coil,:,:,:] = PerformPatternMatchingViaMaxInnerProduct(coilImageData[:,coil,:,:,:], dictionary, simulation)

    means['T1'] = np.mean(patternMatchResults['T1'], axis=0)
    stddevs['T1'] = np.std(patternMatchResults['T1'], axis=0)
    stddevs['T1'] = np.std(patternMatchResults['T1'], axis=0)
    modes['T1'] = scipy.stats.mode(patternMatchResults['T1'], axis=0)
    
    confidence['T1'] = stddevs['T1']

    means['T2'] = np.mean(patternMatchResults['T2'], axis=0)
    stddevs['T2'] = np.std(patternMatchResults['T2'], axis=0)
    confidence['T2'] = stddevs['T2'] 

    return patternMatchResults, M0, means, stddevs, confidence

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

    ## Simulate the Dictionary
    sequence = SequenceParameters("on-the-fly", SequenceType.FISP)
    sequence.Initialize(TRs[:,0]/(1000*1000), TEs[:,0]/(1000*1000), FAs[:,0]/(100), PHs[:,0]/(100), IDs[:,0])
    dictionary = DictionaryParameters.GenerateFixedStep(dictionaryName, fixedStepSize=fixedStepSize, includeB1=includeB1, t1Range=t1Range, t2Range=t2Range, b1Range=b1Range, b1Stepsize=b1Stepsize)
    simulation = Simulation(sequence, dictionary, phaseRange=phaseRange, numSpins=numSpins)
    simulation.Execute(numBatches=numBatches)
    simulation.CalculateSVD(desiredSVDPower=0.995)
    print("Simulated", numSpirals*numSets, "timepoints")
    del simulation.results

    ## Run the Reconstruction
    svdData = ApplySVDCompression(rawdata, simulation, device=torch.device("cpu"))
    (trajectoryBuffer,trajectories,densityBuffer,_) = LoadSpirals(trajectoryFilepath, densityFilepath, numSpirals)
    svdData = ApplyXYZShiftKSpace(svdData, header, acqHeaders, trajectories)
    nufftResults = PerformNUFFTs(svdData, trajectoryBuffer, densityBuffer, matrixSize, matrixSize*2)
    del svdData
    coilImageData = PerformThroughplaneFFT(nufftResults)
    del nufftResults
    imageData, coilmaps = PerformWalshCoilCombination(coilImageData)

    imageMask = GenerateMaskFromCoilmaps(coilmaps)
    patternMatchResults, M0 = PerformPatternMatchingViaMaxInnerProduct(imageData*imageMask, dictionary, simulation)
    
    # Generate Classification Maps from Timecourses and Known Tissue Timecourses

######################################
    # Initial Implementation: GMtimecourse/WMtimecourse/CSFtimecourse are simulated directly 
    ## Simulate the Gray Matter, White Matter, and CSF Timecourses
    tissueDictionary = DictionaryParameters("tissueDictionary", entries=np.array([ WHITE_MATTER_3T[0], GREY_MATTER_3T[0],CSF_3T[0]]))
    tissueSimulation = Simulation(sequence, tissueDictionary, phaseRange=phaseRange, numSpins=numSpins)
    tissueSimulation.Execute()
    tissueSimulation.truncatedResults = torch.matmul(torch.tensor(tissueSimulation.results.transpose()), torch.tensor(simulation.truncationMatrix)).t()

    ## Run for all pixels
    maskedImages = imageData*imageMask
    shape = np.shape(maskedImages)
    print(np.shape(maskedImages))

    timecourses = maskedImages.reshape(shape[0], -1)

    ## Use a pseudoinversion to calculate the alpha/beta/gamma weights
    WM = tissueSimulation.truncatedResults[:,0]; GM = tissueSimulation.truncatedResults[:,1]; CSF = tissueSimulation.truncatedResults[:,2]
    basis = torch.stack([WM, GM, CSF])
    pinv = torch.linalg.pinv(basis)
    coefficients = torch.abs(torch.matmul(timecourses.t().to(torch.cfloat), pinv.to(torch.cfloat))).squeeze()
    sums = torch.sum(coefficients, axis=1).unsqueeze(1)
    normalizedCoefficients = coefficients / sums
    normalizedCoefficients = (torch.round(normalizedCoefficients*100)/100).cpu().numpy()

    wmFractionMap = np.reshape(normalizedCoefficients[:,0], (matrixSize))
    gmFractionMap = np.reshape(normalizedCoefficients[:,1], (matrixSize))
    csfFractionMap = np.reshape(normalizedCoefficients[:,2], (matrixSize))

    print(np.shape(wmFractionMap))
    print(np.shape(patternMatchResults['T1']))

######################################

    images = []
    # Data is ordered [x y z] in patternMatchResults and M0. Need to reorder to [z y x] for ISMRMRD, and split T1/T2 apart
    T1map = patternMatchResults['T1'] * 1000 # to milliseconds
    T1image = PopulateISMRMRDImage(header, T1map, acqHeaders[0,0,0], 0, colormap="FingerprintingT1.pal", window=2500, level=1250)
    images.append(T1image)   
    T2map = patternMatchResults['T2'] * 1000 # to milliseconds
    T2image = PopulateISMRMRDImage(header, T2map, acqHeaders[0,0,0], 1, colormap="FingerprintingT2.pal", window=500, level=200)   
    images.append(T2image)   
 
    M0map = (np.abs(M0) / np.max(np.abs(M0))) * 2**12
    M0image = PopulateISMRMRDImage(header, M0map, acqHeaders[0,0,0], 2)
    images.append(M0image)   

    WMimage = PopulateISMRMRDImage(header, wmFractionMap, acqHeaders[0,0,0], 3)
    images.append(WMimage)   

    GMimage = PopulateISMRMRDImage(header, gmFractionMap, acqHeaders[0,0,0], 4)
    images.append(GMimage)   

    CSFimage = PopulateISMRMRDImage(header, csfFractionMap, acqHeaders[0,0,0], 5)
    images.append(CSFimage)   

    connection.send_image(images)
    connection.send_close()
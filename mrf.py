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

# Folder for debug output files
debugFolder = "/tmp/share/debug"

# Configure dictionary simulation parameters
dictionaryName = "3pctNoB1"
percentStepSize=3; includeB1=False;  t1Range=(0.1,5000); t2Range=(0.1,500); b1Range=None; b1Stepsize=None; #b1Range=(0.1, 2.5); b1Stepsize=0.1; #t1Range=(50,5000); t2Range=(1,500)
phaseRange=(-4*np.pi, 4*np.pi); numSpins=50; numBatches=20
trajectoryFilepath="mrf_dependencies/trajectories/SpiralTraj_FOV250_256_uplimit1916_norm.bin"
densityFilepath="mrf_dependencies/trajectories/DCW_FOV250_256_uplimit1916.bin"
     
# Takes data input as: [cha z y x], [z y x], or [y x]
def PopulateISMRMRDImage(header, data, acquisition, image_index):
    image = ismrmrd.Image.from_array(data.transpose(), acquisition=acquisition, transpose=False)
    image.image_index = image_index

    # Set field of view
    image.field_of_view = (ctypes.c_float(header.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(header.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(header.encoding[0].reconSpace.fieldOfView_mm.z))

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           str(np.max(data)/2),
                         'WindowWidth':            str(np.max(data))})

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
    dictionary = DictionaryParameters.GenerateFixedPercent(dictionaryName, percentStepSize=percentStepSize, includeB1=includeB1, t1Range=t1Range, t2Range=t2Range, b1Range=b1Range, b1Stepsize=b1Stepsize)
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
    
    # Data is ordered [x y z] in patternMatchResults and M0. Need to reorder to [z y x] for ISMRMRD, and split T1/T2 apart
    T1map = np.swapaxes(patternMatchResults['T1'],0,2)
    T1image = PopulateISMRMRDImage(header, T1map, acqHeaders[0,0,0], 1)
    logging.debug("Sending image to client:\n%s", T1image)
    connection.send_image(T1image)
    
    T2map = np.swapaxes(patternMatchResults['T2'],0,2)
    T2image = PopulateISMRMRDImage(header, T2map, acqHeaders[0,0,0], 2)
    logging.debug("Sending image to client:\n%s", T2image)
    connection.send_image(T2image)
    
    M0map = np.abs(np.swapaxes(M0,0,2))
    M0image = PopulateISMRMRDImage(header, M0map, acqHeaders[0,0,0], 3)
    logging.debug("Sending image to client:\n%s", M0image)
    connection.send_image(M0image)
    connection.send_close()
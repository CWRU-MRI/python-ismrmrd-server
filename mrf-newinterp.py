import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import ctypes
import mrdhelper
from datetime import datetime
from pathlib import Path
import shutil

from tqdm import tqdm
from mrftools import *
import sys
from matplotlib import pyplot as plt
import scipy

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt

import os
import pickle
import hashlib

import azureLogging
import time
from connection import Connection
 

# Folder for debug output files
debugFolder = "/tmp/share/debug"
b1Folder = "/usr/share/b1-data"
dictionaryFolder = "/usr/share/dictionary-data"

# Configure dictionary simulation parameters
dictionaryName = "5pct"
percentStepSize=5; includeB1=False;  t1Range=(10,4000); t2Range=(1,500); b1Range=(0.5, 1.55); b1Stepsize=0.05; 
phaseRange=(-np.pi, np.pi); numSpins=15; numBatches=100
trajectoryFilepath="mrf_dependencies/trajectories/SpiralTraj_FOV250_256_uplimit1916_norm.bin"
densityFilepath="mrf_dependencies/trajectories/DCW_FOV250_256_uplimit1916.bin"

# Azure logging configuration (temporary for testing, should be a secret in the cluster not plaintext)
connectionString = ""
tableName = "reconstructionLog"

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
    logging.info(f"K-Space x/y/z shift applied: {x_shift}, {y_shift}, {z_shift}")
    return svdData*torch.complex(x,y)

def vertex_of_parabola(points, clamp=False, min=None, max=None):
    x1 = points[:,0,0]
    y1 = points[:,0,1]
    x2 = points[:,1,0]
    y2 = points[:,1,1]
    x3 = points[:,2,0]
    y3 = points[:,2,1]
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom
    xv = -B / (2*A)
    yv = C - B*B / (4*A)
    if clamp:
        torch.clamp(xv, min, max)
    return (xv, yv)

def GenerateDictionaryLookupTables(dictionaryEntries):
    uniqueT1s = np.unique(dictionaryEntries['T1'])
    uniqueT2s = np.unique(dictionaryEntries['T2'])

    dictionary2DIndexLookupTable = []
    dictionaryEntries2D = np.zeros((len(uniqueT1s), len(uniqueT2s)), dtype=DictionaryEntry)
    dictionary1DIndexLookupTable = np.zeros((len(uniqueT1s), len(uniqueT2s)), dtype=int)
    for dictionaryIndex in tqdm(range(len(dictionaryEntries))):
        entry = dictionaryEntries[dictionaryIndex]
        T1index = np.where(uniqueT1s == entry['T1'])[0]
        T2index = np.where(uniqueT2s == entry['T2'])[0]
        dictionaryEntries2D[T1index, T2index] = entry
        dictionary1DIndexLookupTable[T1index, T2index] = dictionaryIndex
        dictionary2DIndexLookupTable.append([T1index,T2index])
    dictionary2DIndexLookupTable = np.array(dictionary2DIndexLookupTable)
    return uniqueT1s, uniqueT2s, dictionary1DIndexLookupTable, dictionary2DIndexLookupTable


def BatchPatternMatchViaMaxInnerProductWithInterpolation(signalTimecourses, dictionaryEntries, dictionaryEntryTimecourses, voxelsPerBatch=500, device=None, radius=1):
    if(device==None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    with torch.no_grad():

        uniqueT1s, uniqueT2s, dictionary1DIndexLookupTable, dictionary2DIndexLookupTable = GenerateDictionaryLookupTables(dictionaryEntries)

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
                patternValues = dictionaryEntries[maxInnerProductIndices.squeeze().to(torch.long).cpu()].squeeze()
                patternMatches[firstVoxel:lastVoxel] = patternValues
                
                indices = dictionary2DIndexLookupTable[maxInnerProductIndices.squeeze().to(torch.long).cpu()].squeeze()

                numVoxels = len(maxInnerProductIndices)
                neighbor2DIndices = np.reshape(indices.repeat(numNeighbors,axis=1),(np.shape(indices)[0], np.shape(indices)[1],np.shape(offsets)[1], np.shape(offsets)[2])) + offsets
                neighbor2DIndices[:,0,:,:] = np.clip(neighbor2DIndices[:,0,:,:], 0, np.shape(dictionary1DIndexLookupTable)[0]-1)
                neighbor2DIndices[:,1,:,:] = np.clip(neighbor2DIndices[:,1,:,:], 0, np.shape(dictionary1DIndexLookupTable)[1]-1)

                neighborDictionaryIndices = torch.tensor(dictionary1DIndexLookupTable[neighbor2DIndices[:,0,:,:], neighbor2DIndices[:,1,:,:]].reshape(numVoxels, -1)).to(device)
                neighborInnerProducts = torch.take_along_dim(torch.abs(innerProducts),neighborDictionaryIndices,dim=1).squeeze()
                neighborDictionaryEntries = dictionaryEntries[neighborDictionaryIndices.cpu()].squeeze()

                #Sum of inner products through T2 neighbors for each T1 neighbor
                T1InnerProductSums = torch.stack((torch.sum(neighborInnerProducts[:, [0,1,2]], axis=1), torch.sum(neighborInnerProducts[:, [3,4,5]], axis=1), torch.sum(neighborInnerProducts[:,[6,7,8]], axis=1))).t()
                T2InnerProductSums = torch.stack((torch.sum(neighborInnerProducts[:, [0,3,6]], axis=1), torch.sum(neighborInnerProducts[:,[1,4,7]], axis=1), torch.sum(neighborInnerProducts[:,[2,5,8]], axis=1))).t()

                T1s = torch.tensor(neighborDictionaryEntries['T1'][:, [0,3,6]]).to(device)
                stacked_T1 = torch.stack((T1s, T1InnerProductSums))
                stacked_T1 = torch.moveaxis(stacked_T1, 0,1)

                T2s = torch.tensor(neighborDictionaryEntries['T2'][:, [0,1,2]]).to(device)
                stacked_T2 = torch.stack((T2s, T2InnerProductSums))
                stacked_T2 = torch.moveaxis(stacked_T2, 0,1)

                interpolatedValues = np.zeros((numVoxels),dtype=DictionaryEntry)
                interpT1s, _ = vertex_of_parabola(torch.moveaxis(stacked_T1,1,2), clamp=True, min=0, max=np.max(uniqueT1s))
                interpT2s, _ = vertex_of_parabola(torch.moveaxis(stacked_T2,1,2), clamp=True, min=0, max=np.max(uniqueT2s))
                
                interpolatedValues['T1'] = interpT1s.cpu()
                interpolatedValues['T2'] = interpT2s.cpu()
                interpolatedValues['B1'] = 1
                
                # For "edge" voxels, replace the interpolated values with the original pattern matches
                edgeT1s = (indices[:,0] == (len(uniqueT1s)-1)) + (indices[:,0] == (0))
                interpolatedValues[edgeT1s] = patternValues[edgeT1s]
                
                # For "edge" voxels, replace the interpolated values with the original pattern matches
                edgeT2s = (indices[:,1] == (len(uniqueT2s)-1)) + (indices[:,1] == (0))
                interpolatedValues[edgeT2s] = patternValues[edgeT2s]
                
                # For "nan" voxels, replace the interpolated values with the original pattern matches
                nanT1s = np.isnan(interpolatedValues['T1'])
                interpolatedValues[nanT1s] = patternValues[nanT1s]

                # For "nan" voxels, replace the interpolated values with the original pattern matches
                nanT2s = np.isnan(interpolatedValues['T2'])
                interpolatedValues[nanT2s] = patternValues[nanT2s]
                
                interpolatedMatches[firstVoxel:lastVoxel] = interpolatedValues
                pbar.update(1)
                del batchSignals, M0_device, signalNorm_device, simulationNorm_device

        del normalizedSimulationResults, dictionaryEntryTimecourses, dictionaryEntries, signalsTransposed, signalNorm, normalizedSignals, simulationResults
        del simulationNorm
        return patternMatches,interpolatedMatches, M0

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

def LoadB1Map(matrixSize, b1Filename, resampleToMRFMatrixSize=True, deinterleave=True, deleteB1File=True):
    # Using header, generate a unique b1 filename. This is temporary
    try:
        b1Data = np.load(b1Folder + "/" + b1Filename +".npy")
    except:
        logging.info("No B1 map found with requested filename. Trying fallback. ")
        try:
            b1Filename = f"B1Map_fallback"
            b1Data = np.load(b1Folder + "/" + b1Filename +".npy")
        except:
            logging.info("No B1 map found with fallback filename. Skipping B1 correction.")
            return np.array([])

    b1MapSize = np.array(np.shape(b1Data))
    logging.info(f"B1 Input Size: {b1MapSize}")
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
    logging.info(f"B1 Output Size: {np.shape(b1Data)}")
    if(deleteB1File):
        os.remove(b1Folder + "/" + b1Filename +".npy")     
        logging.info(f"Deleted B1 File: {b1Filename}")
    return b1Data
        
def performB1Binning(b1Data, b1Range, b1Stepsize, b1IdentityValue=800):
    b1Bins = np.arange(b1Range[0], b1Range[1], b1Stepsize)
    b1Clipped = np.clip(b1Data, np.min(b1Bins)*b1IdentityValue, np.max(b1Bins)*b1IdentityValue)
    b1Binned = b1Bins[np.digitize(b1Clipped, b1Bins*b1IdentityValue, right=True)]
    logging.info(f"Binned B1 Shape: {np.shape(b1Binned)}")
    return b1Binned

def PatternMatchingViaMaxInnerProductWithInterpolation(combined, dictionary, simulation, voxelsPerBatch=500, b1Binned=None, device=None,):
    if(device==None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    sizes = np.shape(combined)
    numSVDComponents=sizes[0]; matrixSize=sizes[1:4]
    patternMatches = np.empty((matrixSize), dtype=DictionaryEntry)
    interpolatedMatches = np.empty((matrixSize), dtype=DictionaryEntry)
    M0 = torch.zeros((matrixSize), dtype=torch.complex64)
    if b1Binned is not None:
        for uniqueB1 in np.unique(b1Binned):
            logging.info(f"Pattern Matching B1 Value: {uniqueB1}")
            if uniqueB1 == 0:
                patternMatches[b1Binned==uniqueB1] = 0
            else:
                signalTimecourses = combined[:,b1Binned == uniqueB1]
                simulationTimecourses = torch.t(torch.t(torch.tensor(simulation.truncatedResults))[(np.argwhere(dictionary.entries['B1'] == uniqueB1))].squeeze())
                dictionaryEntries = dictionary.entries[(np.argwhere(dictionary.entries['B1'] == uniqueB1))]
                signalTimecourses = combined[:,b1Binned == uniqueB1]
                patternMatches[b1Binned == uniqueB1], interpolatedMatches[b1Binned == uniqueB1], M0[b1Binned == uniqueB1] = BatchPatternMatchViaMaxInnerProductWithInterpolation(signalTimecourses,dictionaryEntries,simulationTimecourses, voxelsPerBatch=voxelsPerBatch, device=device)
    else:
        signalTimecourses = torch.reshape(combined, (numSVDComponents,-1))
        if(dictionary.entries['B1'][0]):
            simulationTimecourses = torch.t(torch.t(torch.tensor(simulation.truncatedResults))[(np.argwhere(dictionary.entries['B1'] == 1))].squeeze())
            dictionaryEntries = dictionary.entries[(np.argwhere(dictionary.entries['B1'] == 1))]
        else:   
            simulationTimecourses = torch.tensor(simulation.truncatedResults)
            dictionaryEntries = dictionary.entries
        patternMatches, interpolatedMatches, M0 = BatchPatternMatchViaMaxInnerProductWithInterpolation(signalTimecourses, dictionaryEntries, simulationTimecourses, voxelsPerBatch=voxelsPerBatch, device=device)
    patternMatches = np.reshape(patternMatches, (matrixSize))
    interpolatedMatches = np.reshape(interpolatedMatches, (matrixSize))
    M0 = np.reshape(M0, (matrixSize)).numpy()
    M0 = np.nan_to_num(M0)
    return patternMatches, interpolatedMatches, M0

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

def GenerateRadialMask(coilImageData, svdNum = 0, angularResolution = 0.01, stepSize = 3, fillSize = 3, maxDecay = 15, featheringKernelSize=4):
    coilMax = np.max(np.abs(coilImageData[svdNum,:,:,:,:].cpu().numpy()), axis=0)
    threshold = np.mean(coilMax)
    maskIm = np.zeros(np.shape(coilMax))
    center = np.array(np.shape(coilMax)[1:3])/2
    Y, X = np.ogrid[:np.shape(coilMax)[2], :np.shape(coilMax)[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    cylindricalMask = dist_from_center <= np.shape(coilMax)[1]/2
    coilMax = cylindricalMask*coilMax
    
    for partition in np.arange(0,np.shape(coilMax)[0]):
        for polarAngle in np.arange(0,2*np.pi, angularResolution):
            decayCounter = 0
            radius = 0
            historicalPos = []
            while decayCounter < maxDecay:
                radius += stepSize
                pos = (center + [radius*np.cos(polarAngle), radius*np.sin(polarAngle)]).astype(int)
                if(pos[0] > 0 and pos[0] < np.shape(coilMax)[1]-1 and pos[1] > 0 and pos[1] < np.shape(coilMax)[2]-1):
                    if coilMax[partition,pos[0],pos[1]] > threshold:
                        for histPos in historicalPos:
                            maskIm[partition, histPos[0]-fillSize:histPos[0]+fillSize, histPos[1]-fillSize:histPos[1]+fillSize] = 1
                        historicalPos.clear()
                        maskIm[partition, pos[0]-fillSize:pos[0]+fillSize, pos[1]-fillSize:pos[1]+fillSize] = 1
                        decayCounter = 0
                    else:
                        decayCounter += 1
                        #maskIm[partition, pos[0]-fillSize:pos[0]+fillSize, pos[1]-fillSize:pos[1]+fillSize] = 1 - (decayCounter/maxDecay)
                        historicalPos.append(pos)
                else:
                     break
    device = torch.device("cpu")
    maskIm = torch.tensor(maskIm).to(torch.float32)  
    meanFilter = torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=featheringKernelSize, bias=False, padding='same')
    featheringKernelWeights = (torch.ones((featheringKernelSize, featheringKernelSize, featheringKernelSize), 
                                          dtype=torch.float32)/(featheringKernelSize*featheringKernelSize*featheringKernelSize)).to(device)
    meanFilter.weight.data = featheringKernelWeights.unsqueeze(0).unsqueeze(0)
    maskIm = meanFilter(maskIm.unsqueeze(0)).squeeze().detach().numpy()
    del featheringKernelWeights, meanFilter
    outputMask = np.moveaxis(maskIm, 0,-1)
    return outputMask

# Generate Classification Maps from Timecourses and Known Tissue Timecourses
def GenerateClassificationMaps(imageData, dictionary, simulation, matrixSize):
    ## Run for all pixels
    shape = np.shape(imageData)
    timecourses = imageData.reshape(shape[0], -1)

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
    T1 = dictionary.entries['T1'][simulation.dictionaryParameters.entries['B1']==1] # Revise to not pass in Dictionary - use the subclass dictionary instead so it matches for sure
    T2 = dictionary.entries['T2'][simulation.dictionaryParameters.entries['B1']==1]
    WmWeights = np.exp( -1 * ( ((T1_wm - T1)/sigmaT1_wm )**2 + ( (T2_wm-T2)/sigmaT2_wm )**2 ))
    GmWeights = np.exp( -1 * ( ((T1_gm - T1)/sigmaT1_gm )**2 + ( (T2_gm-T2)/sigmaT2_gm )**2 )) 
    CsfWeights = np.exp( -1 * ( ((T1_csf - T1)/sigmaT1_csf )**2 + ( (T2_csf-T2)/sigmaT2_csf )**2 )) 

    ## Create timecourses for WM/GM/CSF based on the above 
    truncatedResultsIdentityB1 = simulation.truncatedResults[:,simulation.dictionaryParameters.entries['B1']==1]
    WM = np.sum(truncatedResultsIdentityB1* WmWeights, axis=1); GM = np.sum(truncatedResultsIdentityB1 * GmWeights, axis=1); CSF = np.sum(truncatedResultsIdentityB1 * CsfWeights, axis=1)
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
    
    return (wmFractionMap, gmFractionMap, csfFractionMap)

def ThroughplaneFFT(nufftResults, device=None):
    if(device==None):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    sizes = np.shape(nufftResults)
    numSVDComponents=sizes[0]; numCoils=sizes[1]; numPartitions=sizes[2]; matrixSize=sizes[3:5]
    images = torch.zeros((numSVDComponents, numCoils, numPartitions, matrixSize[0], matrixSize[1]), dtype=torch.complex64)
    with tqdm(total=numSVDComponents) as pbar:
        for svdComponent in np.arange(0, numSVDComponents):
            nufft_device = nufftResults[svdComponent,:,:,:,:].to(device)
            images[svdComponent,:,:,:,:] = torch.fft.ifftshift(torch.fft.ifft(nufft_device, dim=1), dim=1)
            del nufft_device
            pbar.update(1)
    torch.cuda.empty_cache()
    print("Images Shape:", np.shape(images))   
    return images
            
#metadata should be of type ismrmrd.xsd.ismrmrdHeader()
def process(connection:Connection, config, metadata:ismrmrd.xsd.ismrmrdHeader):
    startTime = time.time()

    logging.debug("Config: %s", config)
    header = metadata

    try:
        patientID = metadata.subjectInformation.patientID
        logging.info(f"Patient ID: {patientID}")
        measID = metadata.measurementInformation.measurementID
        measIDComponents = measID.split("_")
        deviceSerialNumber = measIDComponents[0]
        studyID = measIDComponents[2]
        measUID = measIDComponents[3]
        logging.info(f"Device Serial Number: {deviceSerialNumber}")
        logging.info(f"Study ID: {studyID}")
        logging.info(f"Measurement UID: {measUID}")
        b1Filename = f"B1Map_{deviceSerialNumber}_{studyID}"
    except:
        logging.info("Failed to construct B1Map filename. Saving as B1Map_fallback")
        b1Filename = f"B1Map_fallback"


    table_client = azureLogging.setupClient(connectionString, tableName)
    currentScanEntity = azureLogging.setupEntity(table_client, deviceSerialNumber, studyID, measUID)

    enc = metadata.encoding[0]

    numSpirals = enc.encodingLimits.kspace_encoding_step_1.maximum+1; logging.info(f"Spirals: {numSpirals}")
    numMeasuredPartitions = enc.encodingLimits.kspace_encoding_step_2.maximum+1; logging.info(f"Measured Partitions: {numMeasuredPartitions}")
    centerMeasuredPartition = enc.encodingLimits.kspace_encoding_step_2.center; logging.info(f"Center Measured Partition: {centerMeasuredPartition}") # Fix this in stylesheet
    numSets = enc.encodingLimits.set.maximum+1; logging.info(f"Sets: {numSets}")
    numCoils = header.acquisitionSystemInformation.receiverChannels; logging.info(f"Coils: {numCoils}")
    matrixSize = np.array([enc.reconSpace.matrixSize.x,enc.reconSpace.matrixSize.y,enc.reconSpace.matrixSize.z]); logging.info(f"Matrix Size: {matrixSize}")
    numUndersampledPartitions = matrixSize[2]; logging.info(f"Undersampled Partitions: {numUndersampledPartitions}")
    undersamplingRatio = 1
    if(numUndersampledPartitions > 1): # Hack, may not work for multislice 2d
        undersamplingRatio = int(numUndersampledPartitions / (centerMeasuredPartition * 2)); 
        logging.info(f"Undersampling Ratio: {undersamplingRatio}")
    usePartialFourier = False
    if(numMeasuredPartitions*undersamplingRatio < numUndersampledPartitions):
        usePartialFourier = True
        partialFourierRatio = numMeasuredPartitions / (numUndersampledPartitions/undersamplingRatio)
        logging.info(f"Measured partitions is less than expected for undersampling ratio - assuming partial fourier acquisition with ratio: {partialFourierRatio}")
    
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
                FAs[timepoint, measuredPartition] = acq.user_int[4]  # Use requested FA
                #FAs[timepoint, measuredPartition] = acq.user_int[5] # Use actual FA not requested
                PHs[timepoint, measuredPartition] = acq.user_int[6] 
                IDs[timepoint, measuredPartition] = spiral
                acqHeaders[undersampledPartition, spiral, set] = acqHeader
                if rawdata is None:
                    discardPre = int(acqHeader.discard_pre / 2); logging.info(f"Discard Pre: {discardPre}") # Fix doubling in sequence - weird;
                    discardPost = discardPre + 1916; logging.info(f"Discard Post: {discardPost}") # Fix in sequence
                    numReadoutPoints = discardPost-discardPre; logging.info(f"Readout Points: {numReadoutPoints}")
                    rawdata = np.zeros([numCoils, numUndersampledPartitions, numReadoutPoints, numSpirals, numSets], dtype=np.complex64)
                readout = acq.data[:, discardPre:discardPost]
                #readout_fft = torch.fft.fft(torch.tensor(readout), dim=1)
                #(pt_point,_) = torch.max(torch.abs(readout_fft), dim=1)
                #pilotToneData[:, undersampledPartition, spiral, set] = pt_point.numpy() # Change from max to max of
                rawdata[:, undersampledPartition, :, spiral, set] = readout
    except Exception as e:
        logging.exception(e)   
        connection.send_close()   
    
    rawDataTransferFinishTime = time.time()
    azureLogging.updateEntity(table_client, currentScanEntity, 'RawDataTransferTime', rawDataTransferFinishTime-startTime)
    azureLogging.updateEntity(table_client, currentScanEntity, 'AllDataReceived', True)

    (isSavingEnabled,wasSavingSuccessful) = connection.get_dset_save_status()
    if isSavingEnabled and wasSavingSuccessful:
        rawDataFilename = f"MRFData_{deviceSerialNumber}_{studyID}.h5"
        connection.rename_dset_save_file(rawDataFilename)
        try:
            updatedB1Path = os.path.join(connection.savedataFolder,b1Filename +".npy")
            shutil.copyfile(b1Folder + "/" + b1Filename +".npy",updatedB1Path)
        except:
            logging.info("Copying B1 dataset failed")
            return None
        
    rawDataSaveFinishTime = time.time()
    azureLogging.updateEntity(table_client, currentScanEntity, 'RawDataSaveTime', rawDataSaveFinishTime-rawDataTransferFinishTime)
    azureLogging.updateEntity(table_client, currentScanEntity, 'RawDataSaved', True)

    B1map = LoadB1Map(matrixSize, b1Filename)
    if(np.size(B1map)!=0):
        B1map_binned = performB1Binning(B1map, b1Range, b1Stepsize, b1IdentityValue=800)
        dictionary = DictionaryParameters.GenerateFixedPercent(dictionaryName, percentStepSize=percentStepSize, t1Range=t1Range, t2Range=t2Range, includeB1=True, b1Range=b1Range, b1Stepsize=b1Stepsize)
    else:
        B1map_binned = None
        dictionary = DictionaryParameters.GenerateFixedPercent(dictionaryName, percentStepSize=percentStepSize, t1Range=t1Range, t2Range=t2Range, includeB1=False, b1Range=None, b1Stepsize=None)

    if(B1map_binned is not None):
        azureLogging.updateEntity(table_client, currentScanEntity, 'B1CorrectionUsed', True)

    ## Initialize the Sequence
    sequence = SequenceParameters("largescale", SequenceType.FISP)
    sequence.Initialize(TRs[:,0]/(1000*1000), TEs[:,0]/(1000*1000), FAs[:,0]/(100), PHs[:,0]/(100), IDs[:,0])
    simulation = Simulation(sequence, dictionary, phaseRange=phaseRange, numSpins=numSpins)
    simulationHash = hashlib.sha256(pickle.dumps(simulation)).hexdigest()
    dictionaryPath = dictionaryFolder+"/"+simulationHash+".simulation"
    logging.info(f"Dictionary Path: {dictionaryPath}")
    Path(dictionaryFolder).mkdir(parents=True, exist_ok=True)

    azureLogging.updateEntity(table_client, currentScanEntity, 'SimulationHashUsedForReconstruction', simulationHash)

    ## Check if dictionary already exists
    if (os.path.isfile(dictionaryPath)):
        logging.info("Dictionary already exists. Using local copy.")
        filehandler = open(dictionaryPath,'rb')
        simulation = pickle.load(filehandler) 
        filehandler.close()

    else:        
        ## Simulate the Dictionary
        logging.info("Dictionary not found. Simulating. ")
        simulation.Execute(numBatches=numBatches)
        simulation.CalculateSVD(desiredSVDPower=0.98)
        logging.info(f"Simulated {numSpirals*numSets} timepoints")
        del simulation.results
        filehandler = open(dictionaryPath, 'wb')
        pickle.dump(simulation, filehandler)
        filehandler.close()

    ## If dictionary simulation is new, upload to Azure so it will exist forever?

    ## Run the Reconstruction
    svdData = ApplySVDCompression(rawdata, simulation, device=torch.device("cpu"))
    (trajectoryBuffer,trajectories,densityBuffer,_) = LoadSpirals(trajectoryFilepath, densityFilepath, numSpirals)
    svdData = ApplyXYZShift(svdData, header, acqHeaders, trajectories)
    nufftResults = PerformNUFFTs(svdData, trajectoryBuffer, densityBuffer, matrixSize, matrixSize*2)
    del svdData
    coilImageData = ThroughplaneFFT(nufftResults)
    del nufftResults
    imageData, coilmaps = PerformWalshCoilCombination(coilImageData)
    imageMask = GenerateRadialMask(coilImageData)
    patternMatchResults, interpolatedResults, M0 = PatternMatchingViaMaxInnerProductWithInterpolation(imageData, dictionary, simulation, b1Binned = B1map_binned, voxelsPerBatch=2000)
    (wmFractionMap, gmFractionMap, csfFractionMap) = GenerateClassificationMaps(imageData, dictionary, simulation, matrixSize)
    reconstructionFinishTime = time.time()
    azureLogging.updateEntity(table_client, currentScanEntity, 'AllImagesReconstructed', True)
    azureLogging.updateEntity(table_client, currentScanEntity, 'ReconstructionTime', reconstructionFinishTime-rawDataSaveFinishTime)

    ## Send out results
    
    images = []
    # Data is ordered [x y z] in patternMatchResults and M0. Need to reorder to [z y x] for ISMRMRD, and split T1/T2 apart
    #T1map = AddText((imageMask>0.1) * patternMatchResults['T1'] * 1000) # to milliseconds
    #T1image = PopulateISMRMRDImage(header, T1map, acqHeaders[0,0,0], 0, window=2500, level=1250, comment="T1_ms")
    #images.append(T1image) 

    # Data is ordered [x y z] in patternMatchResults and M0. Need to reorder to [z y x] for ISMRMRD, and split T1/T2 apart
    T1map_interp = AddText((imageMask>0.1) * interpolatedResults['T1'] * 1000) # to milliseconds
    T1image_interp = PopulateISMRMRDImage(header, T1map_interp, acqHeaders[0,0,0], 0, window=2500, level=1250, comment="T1_ms")
    images.append(T1image_interp)

    #T2map = AddText((imageMask>0.1) * patternMatchResults['T2'] * 1000) # to milliseconds
    #T2image = PopulateISMRMRDImage(header, T2map, acqHeaders[0,0,0], 2, window=500, level=200, comment="T2_ms")
    #images.append(T2image)   

    # Data is ordered [x y z] in patternMatchResults and M0. Need to reorder to [z y x] for ISMRMRD, and split T1/T2 apart
    T2map_interp = AddText((imageMask>0.1) * interpolatedResults['T2'] * 1000) # to milliseconds
    T2image_interp = PopulateISMRMRDImage(header, T2map_interp, acqHeaders[0,0,0], 1, window=200, level=100, comment="T2_ms")
    images.append(T2image_interp)

    M0map = AddText((imageMask>0.1) * (np.abs(M0) / np.max(np.abs(M0))) * 2**12)
    M0image = PopulateISMRMRDImage(header, M0map, acqHeaders[0,0,0], 2, comment="M0")
    images.append(M0image)   

    #if(np.size(B1map) != 0):
    #    B1image = PopulateISMRMRDImage(header, B1map, acqHeaders[0,0,0], 5, comment="B1")
    #    images.append(B1image)  

    WMimage = PopulateISMRMRDImage(header, AddText((imageMask>0.1) * wmFractionMap), acqHeaders[0,0,0], 3, comment="WM_Fraction")
    images.append(WMimage)   

    GMimage = PopulateISMRMRDImage(header, AddText((imageMask>0.1) * gmFractionMap), acqHeaders[0,0,0], 4, comment="GM_Fraction")
    images.append(GMimage)   

    CSFimage = PopulateISMRMRDImage(header, AddText((imageMask>0.1) * csfFractionMap), acqHeaders[0,0,0], 5, comment="CSF_Fraction")
    images.append(CSFimage)  

    #MASKimage = PopulateISMRMRDImage(header, AddText(imageMask*1000), acqHeaders[0,0,0],9,comment="mask")
    #images.append(MASKimage)

    connection.send_image(images)
    imageTransferFinishTime = time.time()
    azureLogging.updateEntity(table_client, currentScanEntity, 'AllImagesReturnedToScanner', True)
    azureLogging.updateEntity(table_client, currentScanEntity, 'ImageTransferTime', imageTransferFinishTime-reconstructionFinishTime)



    connection.send_close()

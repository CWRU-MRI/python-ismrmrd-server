from azure.data.tables import TableServiceClient
from datetime import datetime
from azure.data.tables import UpdateMode
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def setupClient(connectionString, tableName):
    try:
        return TableServiceClient.from_connection_string(conn_str=connectionString).get_table_client(table_name=tableName)
    except:
        print("Azure Logging Client Setup Failed")
        return None

def setupEntity(table_client: TableServiceClient, deviceSerialNumber, studyID, measUID):
    try:
        currentScanEntity = {
            'PartitionKey': deviceSerialNumber,
            'RowKey': f"{studyID}_{measUID}",
            'DeviceSerialNumber': deviceSerialNumber,
            'StudyID': studyID,
            'MeasurementID': measUID, 
            'AllDataReceived': False, 
            'RawDataSaved': False, 
            'B1CorrectionUsed': False,
            'SimulationHashUsedForReconstruction': '', 
            'AllImagesReconstructed': False, 
            'AllImagesReturnedToScanner': False, 
            'RawDataTransferTime': -1,
            'RawDataSaveTime': -1,
            'ReconstructionTime': -1,
            'ImageTransferTime': -1
        }
        entity = table_client.create_entity(entity=currentScanEntity)
        return currentScanEntity
    except:
        print("Azure Logging Entity Setup Failed")
        return None


def updateEntity(table_client, entity, field, state):
    try:
        toUpdate = table_client.get_entity(partition_key=entity['PartitionKey'], row_key=entity['RowKey'])
        toUpdate[field] = state
        table_client.update_entity(mode=UpdateMode.REPLACE, entity=toUpdate)
    except:
        print("Azure Logging Update Failed")

def uploadFile(connectionString, containerName, deviceSerialNumber, studyID, measUID, filepath):
    blob_service_client = BlobServiceClient.from_connection_string(connectionString)
    blob_client = blob_service_client.get_blob_client(container=containerName, blob=f"{deviceSerialNumber}_{studyID}_{measUID}")
    try:
        with open(file=filepath, mode="rb") as data:
            blob_client.upload_blob(data)
    except:
        print("Azure data upload failed for: ", filepath)

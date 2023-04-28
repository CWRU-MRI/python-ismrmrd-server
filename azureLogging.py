from azure.data.tables import TableServiceClient
from datetime import datetime
from azure.data.tables import UpdateMode

def setupClient(connectionString, tableName):
    return TableServiceClient.from_connection_string(conn_str=connectionString).get_table_client(table_name=tableName)

def setupEntity(table_client: TableServiceClient, deviceSerialNumber, studyID, measUID):
    currentScanEntity = {
        'PartitionKey': deviceSerialNumber,
        'RowKey': studyID,
        'DeviceSerialNumber': deviceSerialNumber,
        'StudyID': studyID,
        'MeasurementID': measUID, 
        'AllDataReceived': False, 
        'B1CorrectionUsed': False,
        'SimulationHashUsedForReconstruction': '', 
        'AllImagesReconstructed': False, 
        'AllImagesReturnedToScanner': False, 
        'RawDataTransferTime': -1,
        'ReconstructionTime': -1,
        'ImageTransferTime': -1
    }
    entity = table_client.create_entity(entity=currentScanEntity)
    return currentScanEntity

def updateEntity(table_client, entity, field, state):
    toUpdate = table_client.get_entity(partition_key=entity['PartitionKey'], row_key=entity['RowKey'])
    toUpdate[field] = state
    table_client.update_entity(mode=UpdateMode.REPLACE, entity=toUpdate)
    
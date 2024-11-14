param containerCount int = 500
param batchSize int = 100
param location string = resourceGroup().location
param containerImage string = 'mcr.microsoft.com/azuredocs/aci-helloworld'

param startupScript string
param experimentId string

@secure()
param dbUrl string

param timeout int
param resourceGroupName string
param subscriptionId string

var numBatches = int(containerCount / batchSize) + (containerCount % batchSize > 0 ? 1 : 0)

module containerBatch 'aciBatch.bicep' = [for i in range(0, numBatches): {
  name: 'containerBatch-${i}'
  params: {
    startIndex: i * batchSize
    endIndex: min((i + 1) * batchSize, containerCount)
    location: location
    containerImage: containerImage
    startupScript: startupScript
    experimentId: experimentId
    dbUrl: dbUrl
    timeout: timeout
    resourceGroupName: resourceGroupName
    subscriptionId: subscriptionId
  }
}]

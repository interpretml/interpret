param startIndex int
param endIndex int
param containerImage string
param location string

param startupScript string
param experimentId string

@secure()
param dbUrl string

param timeout int
param resourceGroupName string
param subscriptionId string

resource containerGroup 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = [for i in range(startIndex, endIndex - startIndex): {
  name: 'containerGroup-${i}'
  location: location
  tags: {
    environment: 'dev'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    containers: [
      {
        name: 'container-${i}'
        properties: {
          image: containerImage
          resources: {
            requests: {
              cpu: 4
              memoryInGB: 16
            }
          }
          environmentVariables: [
            {
              name: 'EXPERIMENT_ID'
              value: experimentId
            }
            {
              name: 'RUNNER_ID'
              value: '${i}'
            }
            {
              name: 'DB_URL'
              secureValue: dbUrl
            }
            {
              name: 'TIMEOUT'
              value: '${timeout}'
            }
            {
              name: 'RESOURCE_GROUP_NAME'
              value: resourceGroupName
            }
            { name: 'CONTAINER_GROUP_NAME'
              value: 'containerGroup-${i}'
            }
            {
              name: 'SUBSCRIPTION_ID'
              value: subscriptionId
            }
          ]
          command: ['/bin/sh', '-c', startupScript]
        }
      }
    ]
    restartPolicy: 'Never'
    osType: 'Linux'
  }
}]

resource roleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for i in range(startIndex, endIndex - startIndex): {
  name: guid(resourceGroup().id, 'containerGroup', '${i}', 'delete-role-assignment')
  scope: resourceGroup()
  properties: {
    principalId: containerGroup[i - startIndex].identity.principalId
    // roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8e3af657-a8ff-443c-a75c-2fe8c4bcb635') // 'Owner' role GUID
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c') // 'Contributor' role GUID
  }
}]

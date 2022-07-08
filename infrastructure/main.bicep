targetScope = 'subscription'

param projectName string
param location string

var randomCode = substring(uniqueString(projectName), 0, 5)
var storageAccountName = 'st${projectName}${randomCode}'
var keyVaultName = 'kv-${projectName}-${randomCode}'
var appInsightsName = 'ai-${projectName}-${randomCode}'
var machineLearningWorkspaceName = 'mlw-${projectName}-${randomCode}'
var containerRegistryName = 'cr${projectName}${randomCode}'

var adminUserIds = []

var tags = {
  Description: 'A description about my project'
}

resource rg_resources 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: 'rg-${projectName}-${randomCode}'
  location: location
  tags: tags
}


module st './modules/storageaccount.bicep' = {
  name: 'storageAccountDeploy'
  scope:rg_resources
  params:{
    name: storageAccountName
    location: location
    sku: 'Standard_LRS'
    kind: 'StorageV2'
    tags: tags
    containers: [
      'documents'
      'data'
    ]
  }
}

module kv 'modules/keyvault.bicep' = {
  name:'keyVaultDeploy'
  scope: rg_resources
  params: {
    location: location
    name: keyVaultName
    tags: tags
    adminUserIds: adminUserIds
  }
}

module ai_web 'modules/applicationinsights.bicep' = {
  name: 'applicationInsightsDeploy'
  scope: rg_resources
  params:{
    location: location
    name: appInsightsName
    type: 'web'
    tags: tags
  }
}

module cr 'modules/containerregistry.bicep' = {
  name: 'containerRegistryDeploy'
  scope: rg_resources
  params:{
    location: location
    name: containerRegistryName
    tags: tags
  }
}

module mlw 'modules/machinelearningworkspace.bicep' = {
  name: 'machinelearningworkspaceDeploy'
  scope: rg_resources
  params:{
    location: location
    name: machineLearningWorkspaceName
    tags: tags
    applicationInsightsId: ai_web.outputs.id
    containerRegistryId: cr.outputs.id
    keyvaultId: kv.outputs.id
    storageAccountId: st.outputs.id
  }
}

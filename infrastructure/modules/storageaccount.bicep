param location string
param name string
param kind string
param sku string
param tags object
param containers array = []

resource st 'Microsoft.Storage/storageAccounts@2021-06-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  kind: kind
  properties:{
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
  }
}

resource container_rs 'Microsoft.Storage/storageAccounts/blobServices/containers@2021-06-01' = [for container in containers:{
  name: '${st.name}/default/${container}'
}]

output staticsite string = st.properties.primaryEndpoints.web
output id string = st.id

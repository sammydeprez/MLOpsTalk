param location string
param name string
param tags object

resource containerRegistryName_resource 'Microsoft.ContainerRegistry/registries@2021-12-01-preview' = {
  tags: tags
  name: name
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
  }
}

output id string = containerRegistryName_resource.id

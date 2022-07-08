param location string
param name string
param tags object

resource search 'Microsoft.Search/searchServices@2020-08-01-preview' = {
  name: 'srch-${name}'
  tags: tags
  location: location
  sku: {
    name: 'basic'
  }
}


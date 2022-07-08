param location string
param name string
param tags object

resource form 'Microsoft.CognitiveServices/accounts@2022-03-01' = {
  name: name
  location: location
  tags: tags
  kind: 'FormRecognizer'
  sku: {
    name: 'S0'
  }
  properties: {
  }
}

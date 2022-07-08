param location string
param name string
param tags object

resource logic 'Microsoft.Logic/workflows@2019-05-01' = {
  name: name
  location: location
  tags: tags
  properties:{
    definition:{
      contentVersion: '1.0.0.0'
      parameters: {}
      actions: {}
      triggers: {}
      outputs: {}
      '$schema': 'https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#'
    }
    parameters: {
    }
    state: 'Enabled'
  }
}

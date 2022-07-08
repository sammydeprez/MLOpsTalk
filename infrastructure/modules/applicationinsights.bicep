param location string
param name string
param type string
param tags object

resource ai 'Microsoft.Insights/components@2020-02-02-preview' = {
  name: name
  location: location
  tags: tags
  kind: type
  properties: {
    Application_Type: type
  }
}

output id string = ai.id
output name string = ai.name
output instrumentationKey string = ai.properties.InstrumentationKey
output connectionstring string = ai.properties.ConnectionString
output appId string = ai.properties.AppId

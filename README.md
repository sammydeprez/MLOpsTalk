## MLOps Talk
The code within this repository gives you a small example how to build & deploy an ML model with the Azure ML v1 SDK.
Note: v2 is in Preview and nothing of this code will work anymore

Slides to the presentation can be found [here](slides/Slides.pdf)

**Project Definition**: Training a classification model to recogize Lego Simpson avatars.

## Step 1: Deployment of the resources on Azure
Big fan of IAC (Infrastructure as Code), therefore I fully scripted the environment that I need to run this project in BICEP.
To be able to execute this code you need to have installed the AZ CLI.
After that run following commands:

```
az login
az account set --subscription <<SUBSCRIPTION_ID>>
az deployment sub create --location <<YOUR PREFERED AZURE REGION>> --template-file infrastructure/main.bicep --parameters projectName='<<PROJECTNAME>>' location='<<YOUR PREFERED AZURE REGION>>'
```

## Step 2: Configure Workspace
After our resources are deployed we want to start configuring our workspace.
Therefore you need to run [01_setup_workspace.ipynb](01_setup_workspace.ipynb)

This Jupyter notebook but any other can run locally or on Azure ML Studio. Note when using the Azure ML Studio you need to configure a Compute Instance

Before you execute the different steps in the notebook make sure you configure the variables at the top of the notebook.

This notebook wil create (if not exists) a compute cluster. This is the compute where the training will happen. Notice we have a minimum node set to 0. Which means that when we do not execute anything, there will be no cost linked to it.

The second step is uploading the images used to train a classification model with pytorch. Except uploading it will also register the dataset in to the Azure ML Studio.

## Step 3: Setup Pipeline
Now that our compute and dataset are ready to use. We can define a pipeline with steps that need to be executed to train our model.

Therefore you need to execute the notebook [02_setup_pipeline.ipynb](02_setup_pipeline.ipynb)

In this notebook we will configure the environment needed to train the model.
Configure the steps needed to train the model
And execute an experiment which will train and register the model into Azure ML Studio

## Step 4: Deployment of the model
Our model is ready so its time to deploy it to an endpoint. In notebook [03_setup_deployment.ipynb](03_setup_deployment.ipynb) we make use of Azure Container Instances to deploy our scoring/inferencing function.

Just as in the training steps we need to define an inferencing environment which we will do by making us of a conda environment.
After that we configure the size of our container instance and deploy the selected model.

## Step 5: Testing the endpoint
At the end of step 4 it will return you an endpoint that you can use.
It will look like this:
```
http://02fc2460-0c10-4e3b-8a8b-0305b3992fd6.westeurope.azurecontainer.io/score
```

In postman or any other REST Client you can execute following command:
```
POST http://02fc2460-0c10-4e3b-8a8b-0305b3992fd6.westeurope.azurecontainer.io/score
Content-Type: application/json

{
    "image":"https://raw.githubusercontent.com/hnky/dataset-lego-figures/master/_test/Bart.jpg"
}
```

As a response you should receive following details:
```
{
  "time": "0.265655",
  "prediction": "Bart-Simpson",
  "scores": {
    "Bart-Simpson": "78.95"
  }
}
```


*Credits to [Henk Boelman (MSFT)](https://github.com/hnky) for the data, training & scoring script.*
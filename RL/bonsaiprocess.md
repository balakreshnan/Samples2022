# Azure Bonsai Reinforcement learning Process

## Steps to show how to use Bonsai to train a model and deploy

## Use Case

- How to best send orders to which manufacturing plant from distribution centers
- Distribution centers takes the orders from their customer
- Goal here is to see who can fulfil in 2 or 3 days time based on quantity available
- This is existing sample

## Prerequisites

- Azure Account
- Azure Storage account
- Azure Bonsai service resource
- Implementation URL - https://github.com/microsoft/bonsai-anylogic/tree/master/samples/product-delivery
- The above URL is a sample RL problem that we are going to explain in this tutorial.

## Steps

### Process Flow

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/RL/Images/rlarch.jpg "Entire Flow")

- Above shows the entire process flow for RL
- This is subjective to change based on new project enhancements
- Also note, the simulation is needed and depends on other 3rd party softwares like anylogic, Mathworks Simulink, VP link

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/RL/Images/rl1.jpg "Entire Flow")

## Training and deploy

- First create a Bonsai workspace resource
- Create a Bonsai project
- Create a brain from this link. Copy and paste
- https://github.com/microsoft/bonsai-anylogic/blob/master/samples/product-delivery/pd_inkling.ink
- The above code is inkling code by bonsai
- Once you have this ready, it's not yet time to train
- Next we need to upload the simulation
- in this example since i dont have any simulation tool i am using the zip file below from this repo
- https://github.com/microsoft/bonsai-anylogic/blob/master/samples/product-delivery/exported.zip
- Wait for the upload to complete
- Then copy the code to package import

```
package "exportedname"
```

- Now go to brain and switch the workspace to show the code
- then search for simAction and paste the package command above
- Now we are ready to train

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/RL/Images/rl2.jpg "Entire Flow")

- Training takes some time and see the iterations
- at one point the iterations will flat line like below

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/RL/Images/rl3.jpg "Entire Flow")

- Now stop the training
- Check the iteration assesments

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/RL/Images/rl5.jpg "Entire Flow")

- Click Export Brain
- This will create a image in container registry

![Entire Flow](https://github.com/balakreshnan/Samples2022/blob/main/RL/Images/rl4.jpg "Entire Flow")

- Now check the exported brains section on the left navigation menu
- There are few ways to deploy the brain
- Pull the image from container registry and deploy in conatiner instance or AKS or local computer
- If you have IoT Edge configure the brain to run in IoT Edge
- Container deploys as REST end point so the input and out put are necessart

- Input

```
requestBody = {
  "state": {
    "acceptingness": 2,
    "num_vehicles": 1.5,
    "production_rates": 0.75,
    "vehicle_utilizations": 0.1,
    "inventory_levels": 1,
    "queue_sizes": 2,
    "rolling_turnaround_hours": 40,
    "accepting_rolling_turnaround_hours": 20,
    "rolling_cost_per_products": 40,
    "accepting_rolling_cost_per_products": 20,
    "time_hours": 4
  }
}
}
```

- Output

```
{
    # Whether each MC should accept orders
    # Orders that would normally go to an MC are redirect to the nearest open MC
    acceptingness: number<0, 1, >[3],

    # Number of vehicles to allocate to each MC
    num_vehicles: number<1 .. 3 step 1>[3],

    # Hourly production rate at each MC
    production_rates: number<50 .. 80>[3]
}
```
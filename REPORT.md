# DRLND-Project1 Report
DRLND Project 1 - Navigation (Banana environment)

## Learning Algorithm

The learning algorithm implemented for this project is a **Double DQN** an optional experimental version of Prioritized Experience Replay.
.
```
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=48, fc2_units=48):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

#### Hyperparameters

```
BUFFER_SIZE = 100000     # replay buffer size
BATCH_SIZE = 64          # minibatch size
DISCOUNT_FACTOR = 0.99   # discount factor
TAU = 0.001              # for soft update of target parameters
LEARNING_RATE = 0.006    # learning rate 
UPDATE_EVERY = 4         # how often to update the network
```

These are unchanged from the Udacity example. The BATCH_SIZE and BUFFER_SIZE are parameters for the **ReplayBuffer** class, an "memory" randomly sampled at each step to obtain _experiences_ passed into the **learn** method with a discount of GAMMA. LEARNING_RATE is a parameter to the **Adam** optimizer. TAU is a parameter for a _soft update_ of the target and local models. Finally, UPDATE_EVERY determines the number of steps before learning from a new sample.

#### Model Architecture

 The model is a mapping of state to action values via fully connected **Linear** layers with **relu** activation. 

## Plot of Rewards

![Training Result](https://github.com/MidasS/udacity_drl_project/blob/master/image002.png)


![Plot of Rewards](https://github.com/MidasS/udacity_drl_project/blob/master/image001.png)


![Test Result](https://github.com/MidasS/udacity_drl_project/blob/master/image003.png)



## Ideas for Future Work

I will apply other DQN and Hyperparameter tuning for better performance.

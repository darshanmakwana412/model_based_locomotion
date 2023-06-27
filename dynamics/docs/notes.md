# NOTES 

If you want to <br />
- know about the env <br />
- change the env <br />
- or make your own env <br />

You might care about the following: <br />

a) from_observation_to_usablestate in data_manipulation.py <br />
- This explains each element of the env's observation
- It also allows you to edit which parts of the observations you want to feed into your NN (this is called "state" throughout the paper)

b) get_indices in data_manipulation.py <br />
- Indicates which index of the state corresponds to what (ex. xindex, yindex, etc.)
- These indeces are used throughout the code, for reward functions/etc.

c) reward_functions.py <br />
- A reward function should be defined for each env/task

---------------------------------------------------------------
---------------------------------------------------------------

### Variables in the yaml files:

**num_rollouts_train** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of rollouts to collect for training dataset<br />

**nEpoch** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of epochs for training the NN dynamics model<br />

**horizon** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; horizon of the MPC controller<br />

**num_control_samples** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of random candidate action sequences generated by MPC controller<br />

**fraction_use_new** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; how much new vs old data to use when training the NN dynamics model<br />

**num_aggregation_iters** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; how many full iterations of training-->rollouts-->aggregatedata to conduct<br />

**num_trajectories_for_aggregation** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; how many MPC rollouts to conduct during each aggregation iteration<br />

**rollouts_forTraining** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; how many of aggregated rollouts to put into training dataset (vs into validation dataset)<br />

**num_fc_layers** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; number of hidden layers in dynamics model<br />

**depth_fc_layers** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dimension of each hidden layer in dynamics model<br />

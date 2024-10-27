# Reinforcement Learning on a Platformer Game

Using the Stable Baselines 3 library, a reinforcement learning library built on PyTorch, I trained an RL agent to play a platformer game I developed (originally in Javascript, later translated to Python).

### learn_and_save

Train a policy with certain hyperparameters and then save the model at certain checkpoints, using TensorBoard to track progress.

### load_and_show

Load a previously saved policy and have it play the game by itself. 

### load_learn_and_save

Load a previously saved policy to train it again, with potentially different hyperparameters.

### main

Play the game yourself! Use arrow keys, WASD, spacebar to jump, or any combination of them.

## Potential future improvements/projects

* Make the game go on forever rather than having a limit but slowly increase the speed of the rising lava until it becomes impossible to escape. Train the agent on this version of the game.
* The original Javascript version of the game included different types of platforms (ones that restricted your jumps, or bounced you like a trampoline, or blocked you from going through from below). It also made the game harder as you went up, making the platforms sparser and sparser. Train the agent on this harder and more complex game.
* Add enemies or obstacles to the game, and train the agent on this version too. Maybe even train the enemies' actions as well, to try to eliminate the player faster.
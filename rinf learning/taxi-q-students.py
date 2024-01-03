import gymnasium as gym
import numpy
import random
import matplotlib.pyplot as plt
from time import sleep

env = gym.make("Taxi-v3").env

# Initialize Q-Table
Qtable = numpy.zeros([env.observation_space.n, env.action_space.n])

learningRate=0.01
discountFactor=0.6
epsilon=0.1

def train(episodes):
    
    global Qtable

    rewards = []

    for i in range(episodes):
        state = env.reset()[0]
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                # get an action
            else:
                # get an action

            nextState, reward, done, truncated, info = env.step(action) 
            
            #Update Q values using using Q learning

            state = nextState
            
        if i % 100 == 0 and i != 0:
            totalReward = evaluate(render=False)
            rewards.append(totalReward)
            print(f"Episode: {i}, max reward so far: {numpy.max(rewards)}")

    print("Training finished.\n")
    return rewards

def plotTrainingData(rewards):
    filename= f"qlearning-lr{learningRate}-g{discountFactor}-e{epsilon}.png"
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward vs. Evaluation Episode')
    plt.savefig(filename)
    plt.show()
    
def saveQTable():
    numpy.save("qTable.npy", Qtable)

def loadQTable():
    global Qtable
    Qtable = numpy.load("qTable.npy")

def evaluate(render=True):
    if render:
        env = gym.make("Taxi-v3", render_mode="human").env
    else:
        env = gym.make("Taxi-v3").env
    
    state = env.reset(seed=20)[0]
    timeStep, reward, totalReward = 0,0,0
    
    maxSteps = 50

    done = False

    while not done and timeStep < maxSteps:
        #Select an action
        
        state, reward, done, truncated, info = env.step(action)

        
        timeStep = timeStep + 1
        
        if render:
            env.render()
            print(f"timeStep: {timeStep}, State: {state}, Action: {action}, Reward: {reward}")
            sleep(0.08)
        
        totalReward += reward
        
    if render:
        env.close()
        
    return totalReward
        
rewards = train(episodes=500)

plotTrainingData(rewards)

evaluate()
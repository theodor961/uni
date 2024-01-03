import gymnasium as gym
import numpy
import random
import matplotlib.pyplot as plt
from time import sleep

env = gym.make("Taxi-v3").env

# Initialize Q-Table
Qtable = numpy.zeros([env.observation_space.n, env.action_space.n])

learningRate=0.1
discountFactor=0.7
epsilon=0.8

def trainQlearning(episodes):
    
    global Qtable

    rewards = []

    for i in range(episodes):
        state = env.reset()[0]
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = numpy.argmax(Qtable[state])

            nextState, reward, done, truncated, info = env.step(action) 
            
            oldQValue = Qtable[state, action]
            maxQValue = numpy.max(Qtable[nextState])

            newQvalue = (1 - learningRate) * oldQValue + learningRate * (reward + discountFactor * maxQValue)
            
            Qtable[state, action] = newQvalue

            state = nextState
            
        if i % 100 == 0 and i != 0:
            totalReward = evaluate(render=False)
            rewards.append(totalReward)
            print(f"Episode: {i}, max reward so far: {numpy.max(rewards)}")

    print("Training finished.\n")
    return rewards

def trainSarsa(episodes):
    global Qtable
    rewards = []

    for i in range(episodes):
        state = env.reset()[0]
        done = False

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(Qtable[state])

        while not done:
            nextState, reward, done, truncated, info = env.step(action)

            if random.uniform(0, 1) < epsilon:
                nextAction = env.action_space.sample()
            else:
                nextAction = numpy.argmax(Qtable[nextState])

            oldQValue = Qtable[state, action]
            nextQValue = Qtable[nextState, nextAction]

            newQvalue = (1 - learningRate) * oldQValue + learningRate * (reward + discountFactor * nextQValue)
            Qtable[state, action] = newQvalue

            state = nextState
            action = nextAction

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
    plt.title('Reward vs. Evaluaiton Episode')
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
        action = numpy.argmax(Qtable[state])
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
        
rewards = trainQlearning(episodes=10000)

saveQTable()

plotTrainingData(rewards)

evaluate()
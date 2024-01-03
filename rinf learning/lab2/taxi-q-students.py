import gymnasium as gym
import numpy
import random
import matplotlib.pyplot as plt
from time import sleep

env = gym.make("Taxi-v3").env

# Initialize Q-Table
Qtable = numpy.zeros([env.observation_space.n, env.action_space.n])

#gamma , ep
learningRate,discountFactor,epsilon, algorithm=0.6,0.1,0.7 , 'qlearning' #'qlearning'|'sarsa'


def train(episodes):
    
    global Qtable

    rewards = []




    for i in range(episodes):
        state = env.reset()[0]
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                # get an action
                action=env.action_space.sample()
            else:
                # get an action
                action=numpy.argmax(Qtable[state])

            nextState, reward, done, truncated, info = env.step(action) 
            
            #Update Q values using using Q learning
           
            if algorithm=='qlearning' :
                    Qtable[state, action] = Qtable[state, action] + learningRate * (reward + discountFactor * numpy.max(Qtable[nextState]) - Qtable[state,action])
            else:
                # Sarsa
                Qtable[state, action] = Qtable[state, action] + learningRate * (reward + discountFactor * Qtable[nextState,action] - Qtable[state,action])


            state = nextState

            
            
        if i % 100 == 0 and i != 0:
            totalReward = evaluate(render=False)
            rewards.append(totalReward)
            print(f"Episode: {i}, max reward so far: {numpy.max(rewards)}")

    print(f"Training finished, using {algorithm} algorithm\n")
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
        action=numpy.argmax(Qtable[state]) 
        
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
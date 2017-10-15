import random
import operator

import numpy as np
import matplotlib.pyplot as plt
import csv as csv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


#########################################################################
# TITANIC INPUT DATA
#########################################################################

train_data=[] # Create a bin to hold our training data.
test_data=[]  # Create a bin to hold our test data.

# Read in CSVs, train and test

with open('train.csv', 'r') as f1:
    f1.readline()
    for row in  csv.reader(f1):       # Skip through each row in the csv file
        train_data.append(row)        # Add each row to the data variable
    train_data = np.array(train_data) # Then convert from a list to a NumPy array

with open('test.csv', 'r') as f2:
    f2.readline()
    for row in csv.reader(f2):      # Skip through each row in the csv file
        test_data.append(row)       # Add each row to the data variable
    test_data = np.array(test_data) # Then convert from a list to an array

# Convert strings to numbers so we can perform computational analysis    
# The gender classifier in column 3: Male = 1, female = 0:
train_data[train_data[0::,3] == 'male', 3] = 1
train_data[train_data[0::,3] == 'female', 3] = 0

# Embark C = 0, S = 1, Q = 2
train_data[train_data[0::,10] == 'C', 10] = 0
train_data[train_data[0::,10] == 'S', 10] = 1
train_data[train_data[0::,10] == 'Q', 10] = 2

# Transfer Null observations
# So where there is no price, I will assume price on median of that class
# Where there is no age I will give median of all ages

# All the ages with no data make the median of the data
train_data[train_data[0::,4] == '',4] = np.median(train_data[train_data[0::,4]\
                                           != '',4].astype(np.float))
# All missing embarks just make them embark from most common place
train_data[train_data[0::,10] == '',10] = np.round(np.mean(train_data[train_data[0::,10]\
                                                   != '',10].astype(np.float)))

train_data = np.delete(train_data,[2,7,9],1) #remove the name data, cabin and ticket
# I need to do the same with the test data now so that the columns are in the same
# as the training data

# I need to convert all strings to integer classifiers:
# male = 1, female = 0:
test_data[test_data[0::,2] == 'male',2] = 1
test_data[test_data[0::,2] == 'female',2] = 0

# Embark C = 0, S = 1, Q = 2
test_data[test_data[0::,9] == 'C',9] = 0 
test_data[test_data[0::,9] == 'S',9] = 1
test_data[test_data[0::,9] =='Q',9] = 2

# All the ages with no data make the median of the data
test_data[test_data[0::,3] == '',3] = np.median(test_data[test_data[0::,3]\
                                           != '',3].astype(np.float))
# All missing embarks just make them embark from most common place
test_data[test_data[0::,9] == '',9] = np.round(np.median(test_data[test_data[0::,9]\
                                                   != '',9].astype(np.float)))
# All the missing prices assume median of their respective class
for i in range(np.size(test_data[0::,0])):
    if test_data[i,7] == '':
        test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') &\
                                             (test_data[0::,0] == test_data[i,0])\
            ,7].astype(np.float))

test_data = np.delete(test_data,[1,6,8],1) # Remove the name data, cabin and ticket

titanic_x = train_data[0::,1::]
titanic_x = titanic_x / np.linalg.norm(titanic_x)
titanic_y = train_data[0::,0]

ind = int(len(titanic_x) * .8)
train_x = titanic_x[:ind]
test_x = titanic_x[ind:]

train_y = titanic_y[:ind]
train_y = titanic_y[ind:]

##############################################################################

creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

def inverse(x):
    return np.power(x, -1)

pset = gp.PrimitiveSet("MAIN", arity=7)
pset.addPrimitive(np.add, arity=2)
pset.addPrimitive(np.subtract, arity=2)
pset.addPrimitive(np.multiply, arity=2)
pset.addPrimitive(np.negative, arity=1)
pset.addPrimitive(inverse, arity=1)
pset.addPrimitive(np.power, arity=2)
pset.addPrimitive(np.maximum, arity=2)
pset.renameArguments(ARG0='Pclass')
pset.renameArguments(ARG1='Sex')
pset.renameArguments(ARG2='Age')
pset.renameArguments(ARG3='SibSp')
pset.renameArguments(ARG4='Parch')
pset.renameArguments(ARG5='Fare')
pset.renameArguments(ARG6='Embarked')

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, x, y, pset):
    func = gp.compile(expr=individual, pset=pset)
    r = []
    for z in x:

        a = func(float(z[0]), float(z[1]), float(z[2]),float(z[3]),float(z[4]),float(z[5]),float(z[6]))
        r.append(a)
    results = [0 if m > 0 else 1 for m in r]
    correct = 0
    for t in zip(results, y):
        if t[0] == int(t[1]):
            correct+=1

    return (len(y)-correct) / len(y)

toolbox.register("evaluate", evalSymbReg, x=titanic_x, y=titanic_y, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

gen = range(40)
avg_list = []
max_list = []
min_list = []

pop = toolbox.population(n=300)

# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = tuple([fit])

# Begin the evolution
for g in gen:
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = tuple([fit])

    # Replace population
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    g_max = max(fits)
    g_min = min(fits)
        
    avg_list.append(mean)
    max_list.append(g_max)
    min_list.append(g_min)

    print("  Min %s" % g_min)
    print("  Max %s" % g_max)
    print("  Avg %s" % mean)
    print("  Std %s" % std)

print("-- End of (successful) evolution --")

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

worst_ind = tools.selWorst(pop, 1)[0]
print("Worst individual is %s, %s" % (worst_ind, worst_ind.fitness.values))

paretoFront = tools.ParetoFront()
paretoFront.update(pop)
print("Pareto Front:")
for p in paretoFront:
    print("%s, %s" % (p, p.fitness.values))

plt.plot(gen, avg_list, label="average")
plt.plot(gen, min_list, label="minimum")
plt.plot(gen, max_list, label="maximum")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc="upper right")
plt.show()
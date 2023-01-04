import matplotlib.pyplot as plt
import numpy as np
import random

#*require package : matplotlib, numpy install using
#pip install matplotlib
#pip install numpy

n_cities = 9
n_population = 100
n_iteration = 500
selectivity = 0.15

################
#list of cities#
################

#longitude 1degree = 111km
coordinates_list = [
    (103.7300402, 1.480024637), (102.566597, 2.033737609), (101.6999833, 3.166665872),
    (113.9845048, 4.399923929), (100.3293679, 5.413613156), (102.2464615, 2.206414407),
    (101.9400203, 2.710492166), (100.3729325, 6.113307718), (102.2299768, 6.119973978)
    ]
name_list = [
    'Johor Bahru','Muar','Kuala Lumpur',
    'Miri','George Town', 'Malacca',
    'Seremban', 'Alor Setar', 'Kota Baharu'
    ]
cities_dict = { x:y for x,y in zip(name_list,coordinates_list)}

layout = "{0:<20}{1:<4}{2:<5}"

print("City list with coordinates (latitude, longitude)")
for key, value in cities_dict.items():
    print(layout.format(str(key), ":", str(value)))
print()

####################################
#Step 1 : Create Initial Population#
####################################
#chromosome/individual
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    #return a particular length list of items, e.g(len()cityList) choosen from the sequence, citylist
    return route

#population
def initialPopulation(popSize, cityList):
    population_set = []

    for i in range(0, popSize):
        population_set.append(createRoute(cityList))

    list_index = []
    for i in range(len(population_set)):
        list_index.append(i)

    popRouteIndex_dict = { x:y for x,y in zip(list_index, population_set)}

    return population_set

def getListIndex(population_set):
    list_index = []
    for i in range(len(population_set)):
        list_index.append(i)

    popRouteIndex_dict = { x:y for x,y in zip(list_index, population_set)}

    return list_index, popRouteIndex_dict

###################################
#Step 2: Compute the Fitness Value#
###################################

def compute_city_distance_coordinates(a,b):
    return (np.sqrt( (a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2)) * 111
    #calculate distance between two city
    # x 111km because of 1 degree (lat, longitude) = 111 km

#passing two cities to calculate the distance
def compute_city_distance_names(city_a, city_b, cities_dict):
    return compute_city_distance_coordinates(cities_dict[city_a], cities_dict[city_b])
    #return the name?/ the distance

#calculate the total distance for one complete route and find the fitness for the route
def fitness_eval(city_list, cities_dict):
    total = 0
    for i in range(n_cities-1):
        a = city_list[i]
        b = city_list[i+1]
        total += compute_city_distance_names(a,b, cities_dict)
    inverseTotal = 1/ total   #the shorter the distance, the higher the fitness
    return inverseTotal

#calculate each of the total inverse distance for the all possigle routes and return in a list
def get_all_fitnes(population_set, cities_dict):
    #original the fitness for all route is 0
    fitnes_list = np.zeros(n_population)

    #Looping over all solutions computing the fitness for each solution
    for i in  range(n_population):
        fitnes_list[i] = fitness_eval(population_set[i], cities_dict)
    return fitnes_list

#######################################################
#Step 3: Selection - Method : Roulette Wheel Selection#
#######################################################

def rouletteWheelSelection(list_index, fitnes_list):
    totalFit = sum(fitnes_list)
    prob_list = []
    for each in fitnes_list:
        prob_fit = each / totalFit
        prob_list.append(prob_fit)

    prob_dict = { x:y for x,y in zip(list_index,prob_list)}

    sorted_prob_dict = dict(sorted(prob_dict.items(), key=lambda x:x[1]))

    cumsum_list = np.cumsum(prob_list)

    sumSelectEachRoute = np.zeros(n_population) #initialization so that all route to be selected is 0
    for i in range(10000):
        temp_prob = random.uniform(0.0, 1)  #uniformly and randomly select probability value between 0-1
        k = 0
        while(temp_prob > 0):
            temp_prob = temp_prob - prob_list[k]
            k+=1
        sumSelectEachRoute[k-1]+=1

    sortedCity = sorted_prob_dict.keys()

    sumSelect_sortedRoute = { x:y for x,y in zip(sumSelectEachRoute,sortedCity) }

    sorted_sumSelect_sortedRoute = dict(reversed(sorted(sumSelect_sortedRoute.items())))

    return sorted_sumSelect_sortedRoute

#select k number of chromosome based on the probability of being selected in roulette wheel selection
def select(list_index, popRouteIndex_dict, fitnes_list, k=4):
    listSelectionParentsIndex = []
    index = 0
    sorted_sumSelect_sortedRoute = rouletteWheelSelection(list_index, fitnes_list)
    for key, bestNRoute in sorted_sumSelect_sortedRoute.items():
        if index < k:
            listSelectionParentsIndex.append(bestNRoute)
            index += 1
        else:
            break

    listSelectionParentsRoute = []
    for index in listSelectionParentsIndex:
        listSelectionParentsRoute.append(popRouteIndex_dict[index])
    return listSelectionParentsRoute

###################
#Step 4: Crossover#
###################
def crossover(listSelectionParentsRoute, p_cross=0.1):
    children = []
    parents = np.asarray(listSelectionParentsRoute)
    count, size = parents.shape
    for _ in range(len(population_set)):
        if np.random.rand() > p_cross:
            children.append(
                list(parents[np.random.randint(count, size=1)[0]])
            )
        else:
            parent1, parent2 = parents[
                np.random.randint(count, size=2), :
            ]
            idx = np.random.choice(range(size), size=2, replace=False)
            start, end = min(idx), max(idx)
            child = [None] * size
            for i in range(start, end + 1, 1):
                child[i] = parent1[i]
            pointer = 0
            for i in range(size):
                if child[i] is None:
                    while parent2[pointer] in child:
                        pointer += 1
                    child[i] = parent2[pointer]
            children.append(child)
    return children

#######################################
#Step 5: Mutation - Inversion Mutation#
#######################################

#swap any two element in chromosome
def swap(chromosome):
    a, b = np.random.choice(len(chromosome), 2)
    chromosome[a], chromosome[b] = (
        chromosome[b],
        chromosome[a],
    )
    return chromosome

###############################
#Step 6: Create New Generation#
###############################

def mutate(listSelectionParentsRoute, p_cross=0.1, p_mut=0.1):
    next_bag = []
    children = crossover(listSelectionParentsRoute, p_cross)
    for child in children:
        if np.random.rand() < p_mut:
            next_bag.append(swap(child))
        else:
            next_bag.append(child)
    return next_bag

#########################################################################################################
#                                         MAIN PROGRAM                                                  #
#########################################################################################################

population_set = initialPopulation(n_population, name_list)
list_index, popRouteIndex_dict = getListIndex(population_set)
fitnes_list = get_all_fitnes(population_set, cities_dict)
listSelectionParentsRoute = select(list_index, popRouteIndex_dict, fitnes_list, n_population*selectivity)
new_pop_set = mutate(listSelectionParentsRoute)

minDistance = 1 / np.max(fitnes_list)
index = np.argmax(np.array(fitnes_list))
bestRoute = popRouteIndex_dict[index]

layout1 = "{0:<5}{1:>5}{2:<5}{3:<10}{4:>5}"
print(layout1.format("Gen", "0", ":", minDistance, "km"))

history = []
history.append(minDistance)

###########################
#Step 7: Stopping Criteria#
###########################

i=1
while (n_iteration > 0):
    prev_minDistance = minDistance
    prev_bestRoute = bestRoute

    list_index, popRouteIndex_dict = getListIndex(new_pop_set)
    fitnes_list = get_all_fitnes(new_pop_set, cities_dict)
    listSelectionParentsRoute = select(list_index, popRouteIndex_dict, fitnes_list, n_population*selectivity)
    new_pop_set = mutate(listSelectionParentsRoute)

    minDistance = 1 / np.max(fitnes_list)
    index = np.argmax(np.array(fitnes_list))
    bestRoute = popRouteIndex_dict[index]

    if(prev_minDistance < minDistance):
        minDistance = prev_minDistance
        bestRoute = prev_bestRoute

    if (i % 100 == 0):
        print(layout1.format("Gen", i, ":", minDistance, "km"))
    history.append(minDistance)

    i = i + 1
    n_iteration = n_iteration - 1


print("Best Route: ",bestRoute)

###########
#Plot path#
###########

point_plot = []
for city in bestRoute:
    coordinates = cities_dict[city]
    point_plot.append(coordinates)
print(point_plot)

for xlongt, ylat in coordinates_list:
    plt.scatter(xlongt, ylat)

plt.xlim(100, 120)
plt.ylim(1,7)
for key, value in cities_dict.items():
    plt.annotate(key, xy=value, xytext=value)
plt.plot(*zip(*point_plot))
plt.grid()
plt.show()

###################
#Optimization Plot#
###################

plt.plot(range(len(history)), history, color="skyblue")
plt.show()


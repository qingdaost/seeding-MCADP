from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
#from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BitFlipMutation, SPXCrossover, IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerCrossover
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.problem.multiobjective.MAD import *#MADT, MADA, MADC, MADM, get_costseed, merge, pool, greedy
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, PrintObjectivesObserver
from jmetal.util.solutions_utils import print_function_values_to_file, print_variables_to_file, get_non_dominated_solutions
from jmetal.util.solutionsgenerator import InjectorGenerator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization.plotting import Plot
from jmetal.core.quality_indicator import *
import numpy as np
import copy
import statistics
import random
import logging

logging.basicConfig(level=logging.INFO, filename="Result20190123.log")

N_seeds = 50
run = 30
fm = 'NSGAII30'


#if __name__ == '__main__':
logging.info(get_apps())
#Find AO seed
problem = MADA()
max_evaluations = 20000
algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=IntegerPolynomialMutation(probability=0.1),
    crossover=IntegerCrossover(probability=0.9),
    selection=BinaryTournamentSelection(),
    termination_criterion=StoppingByEvaluations(max=max_evaluations)
)

#algorithm.observable.register(observer=PrintObjectivesObserver(frequency=10))

algorithm.run()
aggseed = algorithm.get_result()
logging.info('Aggregated seed solution is {}: {}'.format(aggseed.variables, aggseed.objectives[0]))
#print(change)


#Find the min time, also one SO seed
problem = MADT()
max_evaluations = 10000
algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=IntegerPolynomialMutation(probability=0.1),
    crossover=IntegerCrossover(probability=0.9),
    selection=BinaryTournamentSelection(),
    termination_criterion=StoppingByEvaluations(max=max_evaluations),
)
algorithm.run()
sotimeseed = algorithm.get_result()
#mintime = timeseed.objectives[0]
logging.info('SO-time seed solution: {}: {}'.format(sotimeseed.variables, sotimeseed.objectives[0]))

# Find the min cost, also one SO seed
problem = MADC()
#timeseed = problem.create_solution()
# timeseed.variables = greedy()
max_evaluations = 10000
algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    mutation=IntegerPolynomialMutation(probability=0.1),
    crossover=IntegerCrossover(probability=0.9),
    selection=BinaryTournamentSelection(),
    termination_criterion=StoppingByEvaluations(max=max_evaluations),
    # population_generator=InjectorGenerator([timeseed])
)
algorithm.run()
socostseed = algorithm.get_result()
logging.info('SO-cost seed solution: {}: {}'.format(socostseed.variables, socostseed.objectives[0]))

appcostseed = greedy()

# App seeds
#mergeappseed = merge()
apptimeseedset, rateset = pool()
#code, sset = dict()

totalrate = sum(rateset)
f = []
for rate in rateset:
    f.append(rate/totalrate)
#print(f)

#bit-merge

#New
#random:bmr
seedset = [appcostseed] * get_N() + apptimeseedset
#print(seedset)
bmrseeds = []
while len(bmrseeds) <= N_seeds:
    index = random.randrange(get_N() * 2)
    appindex = int(index / 2)
    seed = copy.deepcopy(seedset[index])
    for s in range(get_servicenum()):
        if code[s] not in sset[appindex]:
            another = random.randrange(get_N())
            while code[s] not in sset[another]:
                another = random.randrange(get_N())
            if random.random() > 0.5:
                seed[s] = seedset[another + get_N()][s]
                seed[s + get_servicenum()] = seedset[another + get_N()][s + get_servicenum()]
            else:
                seed[s] = seedset[another + get_N()][s]
                seed[s + get_servicenum()] = seedset[another + get_N()][s + get_servicenum()]
    bmrseeds.append(seed)
#print(bmrseeds)
#weight
bmwseeds = []
while len(bmwseeds) <= N_seeds:
    appindex = np.random.choice(get_N(), 1, p = f)[0]
    if random.random() > 0.5:
        index = appindex
    else:
        index = appindex + get_N()
    seed = copy.deepcopy(seedset[index])
    for s in range(get_servicenum()):
        if code[s] not in sset[appindex]:
            another = random.randrange(get_N())
            while code[s] not in sset[another]:
                another = random.randrange(get_N())
            if random.random() > 0.5:
                seed[s] = seedset[another + get_N()][s]
                seed[s + get_servicenum()] = seedset[another + get_N()][s + get_servicenum()]
            else:
                seed[s] = seedset[another + get_N()][s]
                seed[s + get_servicenum()] = seedset[another + get_N()][s + get_servicenum()]
    bmwseeds.append(seed)
#print(bmwseeds)

#string-merge
#random:smr
smrseeds = seedset
typeset = [appcostseed[:get_servicenum()]] * get_N()
locationset = [appcostseed[get_servicenum():]] * get_N()
for seed in apptimeseedset:
    typeset.append(seed[:get_servicenum()])
    locationset.append(seed[get_servicenum():])
#print(typeset)
while len(smrseeds) <= N_seeds:
    seed = random.choice(typeset) + random.choice(locationset)
    smrseeds.append(seed)
#print(smrseeds)


#weight
smwseeds = seedset
typeset = [appcostseed[:get_servicenum()]] * get_N()
locationset = [appcostseed[get_servicenum():]] * get_N()
for seed in apptimeseedset:
    typeset.append(seed[:get_servicenum()])
    locationset.append(seed[get_servicenum():])
#print(typeset)
while len(smwseeds) <= N_seeds:
    position1 = np.random.choice(get_N(), 1, p=f)[0]
    position2 = np.random.choice(get_N(), 1, p=f)[0]
    typestring = [typeset[position1 * 2], typeset[position1 * 2 + 1]]
    locationstring = [locationset[position2 * 2], locationset[position2 * 2 + 1]]
    seed = random.choice(typestring) + random.choice(locationstring)
    smwseeds.append(seed)
#print(smwseeds)



#Experiments
#No seeds
noseedprocessruns = []
reference = []
#noseedfront = []
for i in range(run):
    problem = MADM()
    seedset = []
    max_evaluations = 100000
    #algorithm = SPEA2(
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=0.1),
        crossover=IntegerCrossover(probability=0.9),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
    )

    # algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    # algorithm.observable.register(observer=VisualizerObserver())

    resultsets = algorithm.run()
    #print(len(resultsets))
    process = []
    for g in range(len(resultsets)):
        process.append(get_non_dominated_solutions(resultsets[g]))

    front = get_non_dominated_solutions(algorithm.get_result())#process[-1]
    process.append(front)
    noseedprocessruns.append(process)
    reference += front


def Experiment(seedstrings):
    problem = MADM()
    seedset = []
    while len(seedset) <= N_seeds:
        for string in seedstrings:
            seed = problem.create_solution()
            seed.variables = string
            seedset.append(seed)

    max_evaluations = 100000
    #algorithm = SPEA2(
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=0.1),
        crossover=IntegerCrossover(probability=0.9),
        termination_criterion=StoppingByEvaluations(max=max_evaluations),
        population_generator=InjectorGenerator(seedset)
    )

    resultsets = algorithm.run()
    process = []
    for g in range(len(resultsets)):
        process.append(get_non_dominated_solutions(resultsets[g]))

    return process, algorithm.get_result()


# Agg
#aggfront = []
aggprocessruns = []
for i in range(run):
    process, final = Experiment([aggseed.variables])

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    aggprocessruns.append(process)
    reference += front


# SO
#sofront = []
soprocessruns = []
for i in range(run):
    process, final = Experiment([sotimeseed.variables, socostseed.variables])

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    soprocessruns.append(process)
    reference += front


# bit-merge-random
#bmrfront = []
bmrprocessruns = []
for i in range(run):
    process, final = Experiment(bmrseeds)

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    bmrprocessruns.append(process)
    reference += front

# bit-merge-weight
#bmwfront = []
bmwprocessruns = []
for i in range(run):
    process, final = Experiment(bmwseeds)

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    bmwprocessruns.append(process)
    reference += front



# sting-merge-random
#smrfront = []
smrprocessruns = []
for i in range(run):
    process, final = Experiment(smrseeds)

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    smrprocessruns.append(process)
    reference += front



# sting-merge-weigh
#smwfront = []
smwprocessruns = []
for i in range(run):
    process, final = Experiment(smwseeds)

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    smwprocessruns.append(process)
    reference += front


reference_front = get_non_dominated_solutions(reference)

plot_front = Plot(plot_title='Pareto front approximation', axis_labels=['TRT', 'TDC'], reference_front=None)
plot_front.plot(reference_front, label=' ', filename=fm, format='png')

# Save results to file
print_function_values_to_file(reference_front, 'FUN.' + algorithm.get_name() + "-" + problem.get_name())
print_variables_to_file(reference_front, 'VAR.' + algorithm.get_name() + "-" + problem.get_name())

# Normalization
timelist = []
costlist = []
for s in reference_front:
    timelist.append(s.objectives[0])
    costlist.append(s.objectives[1])
mintime = min(timelist)
maxtime = max(timelist)
mincost = min(costlist)
maxcost = max(costlist)
print("Time:", mintime, maxtime)
print("Cost:", mincost, maxcost)

normal_reference_front = copy.deepcopy(reference_front)
for s in normal_reference_front:
    s.objectives[0] = (s.objectives[0] - mintime) / (maxtime - mintime)
    s.objectives[1] = (s.objectives[1] - mincost) / (maxcost - mincost)


#Result
#No seeds
HVlist = []
IGDlist = []
for i in range(run):
    HV = []
    IGD = []
    for j in range(len(noseedprocessruns[i])):
        front = noseedprocessruns[i][j]
        for s in front:
            s.objectives[0] = (s.objectives[0] - mintime) / (maxtime - mintime)
            s.objectives[1] = (s.objectives[1] - mincost) / (maxcost - mincost)

        indicator = HyperVolume([1, 1])
        result = indicator.compute(front)
        HV.append(result)

        indicator = InvertedGenerationalDistance(normal_reference_front)
        result = indicator.compute(front)
        IGD.append(result)
    HVlist.append(HV)
    IGDlist.append(IGD)
HVchange = [0] * len(noseedprocessruns[0])
IGDchange = [0] * len(noseedprocessruns[0])
for i in range(len(noseedprocessruns[0])):
    for j in range(run):
        HVchange[i] += HVlist[j][i]
        IGDchange[i] += IGDlist[j][i]
for i in range(len(noseedprocessruns[0])):
    HVchange[i] = HVchange[i] / run
    IGDchange[i] = IGDchange[i] /run

logging.info(HVchange)
logging.info(IGDchange)
logging.info("No seed")
logging.info('HV mean:' + str(HVchange[-1]))
logging.info('IGD mean:' + str(IGDchange[-1]))
logging.info("*" * 20)


#Agg
HVlist = []
IGDlist = []
for i in range(run):
    HV = []
    IGD = []
    for j in range(len(aggprocessruns[i])):
        front = aggprocessruns[i][j]
        for s in front:
            s.objectives[0] = (s.objectives[0] - mintime) / (maxtime - mintime)
            s.objectives[1] = (s.objectives[1] - mincost) / (maxcost - mincost)

        indicator = HyperVolume([1, 1])
        result = indicator.compute(front)
        HV.append(result)

        indicator = InvertedGenerationalDistance(normal_reference_front)
        result = indicator.compute(front)
        IGD.append(result)
    HVlist.append(HV)
    IGDlist.append(IGD)
HVchange = [0] * len(aggprocessruns[0])
IGDchange = [0] * len(aggprocessruns[0])
for i in range(len(aggprocessruns[0])):
    for j in range(run):
        HVchange[i] += HVlist[j][i]
        IGDchange[i] += IGDlist[j][i]
for i in range(len(aggprocessruns[0])):
    HVchange[i] = HVchange[i] / run
    IGDchange[i] = IGDchange[i] /run

logging.info(HVchange)
logging.info(IGDchange)
logging.info("Agg seed")
logging.info('HV mean:' + str(HVchange[-1]))
logging.info('IGD mean:' + str(IGDchange[-1]))
logging.info("*" * 20)


#SO
HVlist = []
IGDlist = []
for i in range(run):
    HV = []
    IGD = []
    for j in range(len(soprocessruns[i])):
        front = soprocessruns[i][j]
        for s in front:
            s.objectives[0] = (s.objectives[0] - mintime) / (maxtime - mintime)
            s.objectives[1] = (s.objectives[1] - mincost) / (maxcost - mincost)

        indicator = HyperVolume([1, 1])
        result = indicator.compute(front)
        HV.append(result)

        indicator = InvertedGenerationalDistance(normal_reference_front)
        result = indicator.compute(front)
        IGD.append(result)
    HVlist.append(HV)
    IGDlist.append(IGD)
HVchange = [0] * len(soprocessruns[0])
IGDchange = [0] * len(soprocessruns[0])
for i in range(len(soprocessruns[0])):
    for j in range(run):
        HVchange[i] += HVlist[j][i]
        IGDchange[i] += IGDlist[j][i]
for i in range(len(soprocessruns[0])):
    HVchange[i] = HVchange[i] / run
    IGDchange[i] = IGDchange[i] /run

logging.info(HVchange)
logging.info(IGDchange)
logging.info("SO seed")
logging.info('HV mean:' + str(HVchange[-1]))
logging.info('IGD mean:' + str(IGDchange[-1]))
logging.info("*" * 20)


#bit-merge-random
HVlist = []
IGDlist = []
for i in range(run):
    HV = []
    IGD = []
    for j in range(len(bmrprocessruns[i])):
        front = bmrprocessruns[i][j]
        for s in front:
            s.objectives[0] = (s.objectives[0] - mintime) / (maxtime - mintime)
            s.objectives[1] = (s.objectives[1] - mincost) / (maxcost - mincost)

        indicator = HyperVolume([1, 1])
        result = indicator.compute(front)
        HV.append(result)

        indicator = InvertedGenerationalDistance(normal_reference_front)
        result = indicator.compute(front)
        IGD.append(result)
    HVlist.append(HV)
    IGDlist.append(IGD)
HVchange = [0] * len(bmrprocessruns[0])
IGDchange = [0] * len(bmrprocessruns[0])
for i in range(len(bmrprocessruns[0])):
    for j in range(run):
        HVchange[i] += HVlist[j][i]
        IGDchange[i] += IGDlist[j][i]
for i in range(len(bmrprocessruns[0])):
    HVchange[i] = HVchange[i] / run
    IGDchange[i] = IGDchange[i] /run

logging.info(HVchange)
logging.info(IGDchange)
logging.info("BMR seed")
logging.info('HV mean:' + str(HVchange[-1]))
logging.info('IGD mean:' + str(IGDchange[-1]))
logging.info("*" * 20)

#bit-merge-weight
HVlist = []
IGDlist = []
for i in range(run):
    HV = []
    IGD = []
    for j in range(len(bmwprocessruns[i])):
        front = bmwprocessruns[i][j]
        for s in front:
            s.objectives[0] = (s.objectives[0] - mintime) / (maxtime - mintime)
            s.objectives[1] = (s.objectives[1] - mincost) / (maxcost - mincost)

        indicator = HyperVolume([1, 1])
        result = indicator.compute(front)
        HV.append(result)

        indicator = InvertedGenerationalDistance(normal_reference_front)
        result = indicator.compute(front)
        IGD.append(result)
    HVlist.append(HV)
    IGDlist.append(IGD)
HVchange = [0] * len(bmwprocessruns[0])
IGDchange = [0] * len(bmwprocessruns[0])
for i in range(len(bmwprocessruns[0])):
    for j in range(run):
        HVchange[i] += HVlist[j][i]
        IGDchange[i] += IGDlist[j][i]
for i in range(len(bmwprocessruns[0])):
    HVchange[i] = HVchange[i] / run
    IGDchange[i] = IGDchange[i] /run

logging.info(HVchange)
logging.info(IGDchange)
logging.info("BMW seed")
logging.info('HV mean:' + str(HVchange[-1]))
logging.info('IGD mean:' + str(IGDchange[-1]))
logging.info("*" * 20)


# sting-merge-random
HVlist = []
IGDlist = []
for i in range(run):
    HV = []
    IGD = []
    for j in range(len(smrprocessruns[i])):
        front = smrprocessruns[i][j]
        for s in front:
            s.objectives[0] = (s.objectives[0] - mintime) / (maxtime - mintime)
            s.objectives[1] = (s.objectives[1] - mincost) / (maxcost - mincost)

        indicator = HyperVolume([1, 1])
        result = indicator.compute(front)
        HV.append(result)

        indicator = InvertedGenerationalDistance(normal_reference_front)
        result = indicator.compute(front)
        IGD.append(result)
    HVlist.append(HV)
    IGDlist.append(IGD)
HVchange = [0] * len(smrprocessruns[0])
IGDchange = [0] * len(smrprocessruns[0])
for i in range(len(smrprocessruns[0])):
    for j in range(run):
        HVchange[i] += HVlist[j][i]
        IGDchange[i] += IGDlist[j][i]
for i in range(len(smrprocessruns[0])):
    HVchange[i] = HVchange[i] / run
    IGDchange[i] = IGDchange[i] /run

logging.info(HVchange)
logging.info(IGDchange)
logging.info("SSR seed")
logging.info('HV mean:' + str(HVchange[-1]))
logging.info('IGD mean:' + str(IGDchange[-1]))
logging.info("*" * 20)


#string-merge-weigh
HVlist = []
IGDlist = []
for i in range(run):
    HV = []
    IGD = []
    for j in range(len(smwprocessruns[i])):
        front = smwprocessruns[i][j]
        for s in front:
            s.objectives[0] = (s.objectives[0] - mintime) / (maxtime - mintime)
            s.objectives[1] = (s.objectives[1] - mincost) / (maxcost - mincost)

        indicator = HyperVolume([1, 1])
        result = indicator.compute(front)
        HV.append(result)

        indicator = InvertedGenerationalDistance(normal_reference_front)
        result = indicator.compute(front)
        IGD.append(result)
    HVlist.append(HV)
    IGDlist.append(IGD)
HVchange = [0] * len(smwprocessruns[0])
IGDchange = [0] * len(smwprocessruns[0])
for i in range(len(smwprocessruns[0])):
    for j in range(run):
        HVchange[i] += HVlist[j][i]
        IGDchange[i] += IGDlist[j][i]
for i in range(len(smwprocessruns[0])):
    HVchange[i] = HVchange[i] / run
    IGDchange[i] = IGDchange[i] /run

logging.info(HVchange)
logging.info(IGDchange)
logging.info("SSW seed")
logging.info('HV mean:' + str(HVchange[-1]))
logging.info('IGD mean:' + str(IGDchange[-1]))
logging.info("*" * 20)
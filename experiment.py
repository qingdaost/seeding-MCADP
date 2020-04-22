from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
#from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BitFlipMutation, SPXCrossover, IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerCrossover
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.problem.multiobjective.MAD import *
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
fm = 'NSGAII'


#if __name__ == '__main__':
logging.info(get_apps())
#Find AO seed: GA
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


#Find the min time, also SO time seed: GA
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

# Find the min cost, also SO cost seed: GA
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


# application-specific seeds
appcostseed = greedy()
apptimeseedset, rateset = pool()

#BB solutions
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


#Experiments
#No seeds: MOEAs
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

#seeding MOEAs
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


# AO-Seed
#aggfront = []
aggprocessruns = []
for i in range(run):
    process, final = Experiment([aggseed.variables])

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    aggprocessruns.append(process)
    reference += front


# SO-Seed
#sofront = []
soprocessruns = []
for i in range(run):
    process, final = Experiment([sotimeseed.variables, socostseed.variables])

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    soprocessruns.append(process)
    reference += front



# BB-Seed
#smrfront = []
smrprocessruns = []
for i in range(run):
    process, final = Experiment(smrseeds)

    front = get_non_dominated_solutions(final)#process[-1]
    process.append(front)
    smrprocessruns.append(process)
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


#AO-Seed
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
#get sd
logging.info('HV final:' + str(HVfinal))
logging.info('IGD final:' + str(IGDfinal))
logging.info("*" * 20)


#SO-Seed
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
#get sd
logging.info('HV final:' + str(HVfinal))
logging.info('IGD final:' + str(IGDfinal))
logging.info("*" * 20)


#BB-Seed
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
#get sd
logging.info('HV final:' + str(HVfinal))
logging.info('IGD final:' + str(IGDfinal))
logging.info("*" * 20)

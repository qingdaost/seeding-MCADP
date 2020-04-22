import random
from math import sqrt, exp, pow, sin

from jmetal.core.problem import FloatProblem, BinaryProblem, IntegerProblem
from jmetal.core.solution import FloatSolution, BinarySolution, IntegerSolution
#import experiment as ex

import numpy as np
import helper
from numpy import random
import copy


random.seed(0)

latency = helper.latencymap
cloud = helper.datacenter
requestdata = helper.request

workflowset = helper.wset
sdict = helper.sdict

capacity = [1, 2, 4, 8, 16, 48, 64, 96]
#number of composite applications
N_WORKFLOW = 3

def get_N():
    return N_WORKFLOW

#generate workflow set
sample = set()
while len(sample) < N_WORKFLOW:
    sample.add(random.randint(10))
    #sample.append(i)
print(sample)

def get_apps():
    return sample

wset = []
for i in sample:
    wset.append(workflowset[i])
print(wset)

#extract servieces
def getsset(workflow):
    sset = set()
    for i in workflow:
        sset.add(i[2])
    return sset

sset = []
for i in range(N_WORKFLOW):
    sset.append(getsset(wset[i]))

print(sset)

#generate request rates
def requestdistribution(request):
    distribution = request[:]
    temp = set()
    while len(temp) < 72:
        index = random.randint(82)
        if index not in temp:
            distribution[index] = 0
            temp.add(index)
    return distribution

requestset = []
rateset = []
for i in range(N_WORKFLOW):
    request = requestdistribution(requestdata)
    requestset.append(request)
    r = sum(request)
    rateset.append(r)
    print(i, ":", r)


#Merge workflows
T = {}
for i in range(N_WORKFLOW):
    for n in wset[i]:
        T[n[2]] = T.get(n[2], 0) + rateset[i]
print("request list (T):", T)
DNA_SIZE = len(T)

code = []
temp = list(sdict.keys())
for i in range(len(sdict)):
    if temp[i] in T.keys():
        code.append(temp[i])
print('CODE:', code)

def get_servicenum():
    return len(code)

#get network latency
def getnl(request, workflow, solution):
    start = cloud[solution[1][code.index(workflow[0][2])]]
    end = cloud[solution[1][code.index(workflow[-1][2])]]
    nl = 0
    for j in range(82):
        nl += request[j] * latency[j][start[0]]
    anl = nl / sum(request)

    nl = 0
    for j in range(82):
        nl += request[j] * latency[j][end[0]]
    anl += nl / sum(request)
    # print("network latency: ", anl)
    return anl

#get makespan
def getwrt(workflow, r, solution):
    estlist = []
    for i in workflow:
        est = 0
        if i[0] != []:
            for j in i[0]:
                s = workflow[j - 1][2]
                if 1000 / sdict[s] * capacity[solution[0][code.index(s)]] > r:
                    temp = estlist[j - 1] + 1 / (
                            1000 / sdict[s] * capacity[solution[0][code.index(s)]] - r) * 1000 + \
                           latency[cloud[solution[1][code.index(s)]][0]][cloud[solution[1][code.index(i[2])]][0]]
                else:
                    temp = float("inf")
                if temp > est:
                    est = temp
        estlist.append(est)
    # print(estlist)
    s = workflow[-1][2]
    if 1000 / sdict[s] * capacity[solution[0][code.index(s)]] > r:
        rt = estlist[-1] + 1 / (1000 / sdict[s] * capacity[solution[0][code.index(s)]] - r) * 1000
    else:
        rt = float("inf")
    # print("##response time: ", rt)
    return rt

#get response time
def get_rt(solution):
    count = 0
    for i in range(N_WORKFLOW):
        wrt = getwrt(wset[i], rateset[i], solution)
        # print(wrt)
        nl = getnl(requestset[i], wset[i], solution)
        # print(nl)
        count += (wrt + nl)  * rateset[i]
    art = count / sum(rateset)
    return art


# get cost
def get_cost(ind):
    cost = 0
    for i in range(len(T)):
        # print(i)
        cost += cloud[ind[1][i]][2][ind[0][i]]
    return cost


def get_costseed():
    types = []
    for s in code:
        r = T[s]
        for c in range(8):
            if 1000 / sdict[s] * capacity[c] > r:
                break
        types.append(c)
    return types + len(code) * [0] #, get_cost([types, len(code) * [0]]), get_cost([types, len(code) * [12]]))

def greedy():
    '''
    min = 10000
    for i in range(15):
        ind = [len(code) * [7], len(code) * [i]]
        temp = get_rt(ind)
        if temp < min:
            min = temp
            l = i
    timeseed = len(code) * [7] + len(code) * [l]
    print("Time seed:", timeseed, get_rt([len(code) * [7], len(code) * [l]]))
    '''
    types = []
    for s in code:
        r = T[s]
        for c in range(8):
            if 1000 / sdict[s] * capacity[c] > r:
                break
        types.append(c)
    costseed = types + len(code) * [0]
    print("Cost seed:", costseed, get_cost([types, len(code) * [0]]))
    return costseed

def pool():
    seedset = []
    for n in range(N_WORKFLOW):
        #seed = ind.variables
        flag = 0
        minimum = float("inf")
        for i in cloud:
            nl = 0
            for j in range(82):
                nl += requestset[n][j] * latency[j][i[0]]
            # print(nl / rateset[n])
            if nl < minimum:
                minimum = nl
                k = flag
            flag += 1
        seed = len(code) * [7] + len(code) * [k]
        print(n, "appseed:", seed)
        seedset.append(seed)
    return seedset, rateset

def dict():
    return code, sset

def candidate(cost, time):
    bitsset = []
    for s in range(len(code)):
        bitset = []
        for n in range(N_WORKFLOW):
            #if code[s] in sset[n]:
            bitset.append([cost[n][s], cost[n][s + len(code)]])
            bitset.append([time[n][s], time[n][s + len(code)]])
        bitsset.append(bitset)
    return bitsset


#AO-Seed
class MADA(IntegerProblem):

    def __init__(self):
        super(MADA, self).__init__()
        self.number_of_variables = len(code) * 2
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        #self.lower_bound = self.number_of_variables * [0]
        types = []
        for s in code:
            r = T[s]
            for c in range(8):
                if 1000 / sdict[s] * capacity[c] > r:
                    break
            types.append(c)
        self.lower_bound = types + len(code) * [0]
        self.upper_bound = len(code) * [7] + len(code) * [14]

        IntegerSolution.lower_bound = self.lower_bound
        IntegerSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        string = solution.variables
        types = string[:len(code)]
        locations = string[len(code):]
        ind = [types, locations]

        f1 = get_rt(ind)
        f2 = get_cost(ind)

        solution.objectives[0] = f1 / 1000 + f2 / get_cost([len(code) * [7], len(code) * [12]])
        #solution.objectives[1] = f2

        return solution

    def get_name(self):
        return 'MADA'

#SO-Seed for time
class MADT(IntegerProblem):

    def __init__(self):
        super(MADT, self).__init__()
        self.number_of_variables = len(code) * 2
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        #self.lower_bound = self.number_of_variables * [0]
        types = []
        for s in code:
            r = T[s]
            for c in range(8):
                if 1000 / sdict[s] * capacity[c] > r:
                    break
            types.append(c)
        self.lower_bound = types + len(code) * [0]
        self.upper_bound = len(code) * [7] + len(code) * [14]

        IntegerSolution.lower_bound = self.lower_bound
        IntegerSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        string = solution.variables
        types = string[:len(code)]
        locations = string[len(code):]
        ind = [types, locations]

        f1 = get_rt(ind)
        #f2 = get_cost(ind)

        solution.objectives[0] = f1
        #solution.objectives[1] = f2

        return solution

    def get_name(self):
        return 'MADT'


#SO-Seed for cost
class MADC(IntegerProblem):

    def __init__(self):
        super(MADC, self).__init__()
        self.number_of_variables = len(code) * 2
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        #self.lower_bound = self.number_of_variables * [0]
        types = []
        for s in code:
            r = T[s]
            for c in range(8):
                if 1000 / sdict[s] * capacity[c] > r:
                    break
            types.append(c)
        self.lower_bound = types + len(code) * [0]
        self.upper_bound = len(code) * [7] + len(code) * [14]

        IntegerSolution.lower_bound = self.lower_bound
        IntegerSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        string = solution.variables
        types = string[:len(code)]
        locations = string[len(code):]
        ind = [types, locations]

        #f1 = get_rt(ind)
        f2 = get_cost(ind)

        #solution.objectives[0] = f1
        solution.objectives[0] = f2

        return solution

    def get_name(self):
        return 'MADC'

#MOP
class MADM(IntegerProblem):

    def __init__(self):
        super(MADM, self).__init__()
        self.number_of_variables = len(code) * 2
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        #self.lower_bound = self.number_of_variables * [0]
        types = []
        for s in code:
            r = T[s]
            for c in range(8):
                if 1000 / sdict[s] * capacity[c] > r:
                    break
            types.append(c)
        self.lower_bound = types + len(code) * [0]
        self.upper_bound = len(code) * [7] + len(code) * [14]

        IntegerSolution.lower_bound = self.lower_bound
        IntegerSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        string = solution.variables
        types = string[:len(code)]
        locations = string[len(code):]
        ind = [types, locations]
        #print(len(solution.objectives))

        f1 = get_rt(ind)
        f2 = get_cost(ind)

        solution.objectives[0] = f1
        solution.objectives[1] = f2

        return solution

    def get_name(self):
        return 'MADM'

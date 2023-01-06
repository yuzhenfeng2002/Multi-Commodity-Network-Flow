"""
Multi Commodity Network Flow
Using Dantzig-Wolfe Decomposition Algorithm
Copyright (c) 2023 Yuzhen FENG
"""
import heapq
import time
import numpy as np
import pandas as pd
import gurobipy as gp

class Node:
    """
    Node in the graph
    """
    def __init__(self, id: int, x=None, y=None) -> None:
        self.id = id
        ## Position
        self.x = x
        self.y = y
        ## Relationship with other nodes
        self.succ = {} # successors in the graph
        self.pred = {} # predecessors in the graph
        self.label = np.inf # for Dijkstra algorithm
        self.spPred = None # also for Dijkstra algorithm
    def __str__(self) -> str:
        return str(self.id)
    def __repr__(self) -> str:
        return str(self.id)
    def __lt__(self, other):
        return self.id < other.id
    
    ## Add successor or predecessor
    def addSucc(self, eId: int, succId: int):
        self.succ[succId] = eId
    def addPred(self, eId: int, predId: int):
        self.pred[predId] = eId

class Edge:
    """
    Edge in the graph
    """
    def __init__(self, id: int, u: Node, v: Node, weight: float, capacity: float = np.inf) -> None:
        self.id = id
        ## The edge is from Node u to Node v
        self.u = u
        self.u.addSucc(id, v.id)
        self.v = v
        self.v.addPred(id, u.id)

        self.capacity = capacity # capacity
        self.weight = weight # weight, may change in the future
        self.initWeight = weight # record the initial weight
        self.capacityUsed = 0 # capacity that has been used by the current flow
        self.capacityLeft = capacity # capacity not used
    
    def __str__(self) -> str:
        return "{:.0f}: {:.0f}->{:.0f} ({:.2f}, {:.2f})".format(self.id, self.u.id, self.v.id, self.initWeight, self.capacity)
    def __repr__(self) -> str:
        return "{:.0f}: {:.0f}->{:.0f} ({:.2f}, {:.2f})".format(self.id, self.u.id, self.v.id, self.initWeight, self.capacity)
    
    ## Update capacity by using a given quantity
    def useCapacity(self, capacityUsed: float):
        self.capacityUsed += capacityUsed
        self.capacityLeft = self.capacityLeft - capacityUsed

class Demand:
    """
    Demand in the graph
    """
    def __init__(self, id: int, o: Node, d: Node, quantity: float) -> None:
        self.id = id
        self.o = o
        self.d = d
        self.quantity = quantity # the quantity of the flow
        self.routes = {} # store the resulting routes of the MCNF problem
    
    ## Hash the route to a string
    def hashRoute(self, route: list):
        s = ""
        for node in route:
            node: Node
            s += str(node.id) + "->"
        return s

    ## Update the dict of the routes, where route is a list of Nodes and ratio is the ratio of flow on the route
    def updateRoute(self, route: list, ratio: float):
        self.routes[self.hashRoute(route)] = self.routes.setdefault(self.hashRoute(route), 0) + ratio

class Network:
    """
    The network
    """
    def __init__(self, nodes={}, edges={}, demands={}) -> None:
        self.nodes: dict = nodes
        self.edges: dict = edges
        self.demands: dict = demands
        self.edgeDict: dict = {} # map the pair of nodes' id to the edge's id
        for edgeId, edge in self.edges.items():
            edge: Edge
            u = edge.u
            v = edge.v
            self.edgeDict[(u.id, v.id)] = edge.id

    ## Add a node
    def addNode(self, id: int, x=None, y=None):
        node = Node(id, x, y)
        self.nodes[id] = node
    
    ## Add an edge, if the end nodes of the edge do not exit, it will add them
    def addEdge(self, id: int, uid: int, vid: int, weight: float, capacity: float = np.inf):
        if uid not in self.nodes:
            self.addNode(uid)
        if vid not in self.nodes:
            self.addNode(vid)
        u = self.nodes[uid]
        v = self.nodes[vid]
        edge = Edge(id, u, v, weight, capacity)
        self.edges[id] = edge
        self.edgeDict[(uid, vid)] = id
    
    ## Add a demand
    def addDemand(self, id: int, oid: int, did: int, quantity: float = 0):
        o = self.nodes[oid]
        d = self.nodes[did]
        demand = Demand(id, o, d, quantity)
        self.demands[id] = demand
    
    ## Load network and demand from a file, example file shown in ./network/*
    def loadNetwork(self, networkFileName: str = "./network/SiouxFalls_net.csv", demandFileName: str = "./network/SiouxFalls_trips.csv"):
        network = pd.read_csv(networkFileName, sep='\t')
        networkDf = network.to_dict("index")
        for edgeId, edgeInfo in networkDf.items():
            self.addEdge(edgeId, edgeInfo['init_node'], edgeInfo['term_node'], edgeInfo['length'], edgeInfo['capacity'] * 2)
        demand = pd.read_csv(demandFileName, sep='\t')
        demandDf = demand.to_dict("index")
        for demandId, demandInfo in demandDf.items():
            if demandInfo["demand"] > 1e-10:
                self.addDemand(demandId, demandInfo["init_node"], demandInfo["term_node"], demandInfo["demand"])

    ## Reset the labels and spPred of all nodes (for new executions of shortest path algorithm)
    def resetNodeLabel(self):
        for node in self.nodes.values():
            node: Node
            node.label = np.inf
            node.spPred = None
    
    ## Reset the capacity of all edges to the initial state
    def resetEdgeCapacity(self):
        for edge in self.edges.values():
            edge: Edge
            edge.capacityUsed = 0
            edge.capacityLeft = edge.capacity
    
    ## Reset the weight of all edges to the initial value
    def resetEdgeWeight(self):
        for edge in self.edges.values():
            edge: Edge
            edge.weight = edge.initWeight

    ## Dijkstra algorithm, return the list of Nodes from u to v and the shortest distance
    def dijkstra(self, u: Node, v: Node = None):
        self.resetNodeLabel()
        self.nodes[u.id].label = 0
        self.nodes[u.id].spPred = None
        minHeap = [(u.label, u)]
        while minHeap:
            currentNode = heapq.heappop(minHeap)[1]
            currentLabel = currentNode.label
            if v != None:
                if v.id == currentNode.id:
                    path = [v]
                    spPred: Node = v.spPred
                    while spPred != None:
                        path.append(spPred)
                        spPred = spPred.spPred
                    path.reverse()
                    return path, v.label
            for nodeId, edgeId in currentNode.succ.items():
                succNode: Node = self.nodes[nodeId]
                succEdge: Edge = self.edges[edgeId]
                if currentLabel + succEdge.weight < succNode.label:
                    succNode.label = currentLabel + succEdge.weight
                    succNode.spPred = currentNode
                    heapq.heappush(minHeap, (succNode.label, succNode))
    
    ## For each demand, assign all the flow to the current shortest path to generate an extreme solution (i.e., a column in DW formulation)
    def generatePathForDemand(self):
        # the information of the extreme solution: 储存该极点的信息
        solution = {"routes": {}}
        # total travel cost of the flow of the solution: 总成本（通行成本）
        totalCost = 0
        # reduced cost of the column: 检验数
        reducedCost = 0
        ## For each demand, assign all the flow to the current shortest path: 对于每一个OD对，计算在调整权重后的路网上的最短路径，并分配流量
        for demandId, demand in self.demands.items():
            demand: Demand
            # calculate the shortest path: 得到最短路径
            sp, _ = self.dijkstra(demand.o, demand.d)
            # the information of the extreme solution -- route: 极点信息——路径
            solution["routes"][demandId] = sp
            # add the total travel cost and the reduced cost according to the path: 根据路径计算总成本和检验数
            lastNode = demand.o
            for node in sp[1: ]:
                node: Node
                # find the edge according to the id of end nodes: 根据路径列表中的点找到对应的边
                edgeId = self.edgeDict[(lastNode.id, node.id)]
                edge: Edge = self.edges[edgeId]
                # update the capacity of the edge: 改变边上的流量
                edge.useCapacity(demand.quantity)
                # total travel cost: 总成本
                totalCost += demand.quantity * edge.initWeight
                # reduced cost: 检验数
                reducedCost += demand.quantity * edge.weight
                lastNode = node
        # the information of the extreme solution: 极点信息
        solution["cost"] = totalCost # total travel cost: 总成本
        solution["reducedCost"] = reducedCost # reduced cost: 检验数（只有这一次迭代有用）
        solution["flow"] = {} # flow in each edge: 各边上的流量
        ## Calculate flow in each edge: 计算各边上的流量
        for edgeId, edge in self.edges.items():
            edge: Edge
            solution["flow"][edgeId] = edge.capacityUsed
        ## Return the extreme solution as well as the information: 返回新极点（及其信息）
        return solution
    
    ## Solve the subproblem
    def subproblem(self, dualVars: list):
        ## Set the weight of all edges according to the dual: 调整路网上各边的权重
        i = 0
        for edgeId, edge in self.boundedEdges.items():
            edge: Edge
            edge.weight = edge.initWeight - dualVars[i]
            i += 1
        ## Reset the capacity of all edges: 重置图上各边的流量
        self.resetEdgeCapacity()
        ## Obtain a new extreme solution as well as its information: 得到一个新极点（对应一列）
        solution = self.generatePathForDemand()
        ## Calculate the reduced cost (add dual associated with sum(lambda)=1): 计算检验数
        reducedCost = -solution["reducedCost"] + dualVars[i]
        ## If the reduced cost is positive, add the solution to the list of columns: 如果检验数为正，就将该极点添加到极点列表中
        if reducedCost > 0:
            self.solutions.append(solution)
        ## Return the reduced cost: 返回检验数
        return reducedCost
    
    ## Process the final solution of the total model
    def retrieveSol(self, output=0, outputFilefolder="./output/"):
        ## Reset capacity and weight of all edges
        self.resetEdgeCapacity()
        self.resetEdgeWeight()
        ## Obtain the lambda
        lams = {}
        for v in self.objModel.getVars():
            if v.X != 0:
                varName = v.VarName
                varValue = v.X
                if len(varName) > 3 and varName[:3] == "lam":
                    lamIdx = int(varName[4:-1])
                    lams[lamIdx] = varValue
        ## For each demand, obtain its route(s) and the ratio of flow on the route according to lambda
        for solId, ratio in lams.items():
            solution = self.solutions[solId]
            routesDict: dict = solution["routes"]
            for demandId, route in routesDict.items():
                demand: Demand = self.demands[demandId]
                demand.updateRoute(route, ratio)
                lastNode: Node = demand.o
                for node in route[1:]:
                    node: Node
                    edgeId = self.edgeDict[(lastNode.id, node.id)]
                    edge: Edge = self.edges[edgeId]
                    edge.useCapacity(demand.quantity * ratio)
                    lastNode = node
        ## Output
        routeFile = open(outputFilefolder+"routes.txt", 'w')
        routeFile.write("id\tratio\troute\n")
        edgeFile = open(outputFilefolder+"flow.txt", 'w')
        edgeFile.write("id\tuid\tvid\tflow\tcapacity\n")
        for demand in self.demands.values():
            demand: Demand
            for route, ratio in demand.routes.items():
                if output: print("{:.0f}\t{:.6f}\t".format(demand.id, ratio) + route)
                else: routeFile.write("{:.0f}\t{:.6f}\t".format(demand.id, ratio) + route + '\n')
        for edge in self.edges.values():
            edge: Edge
            if output: print("{:.0f}\t{:.1f}/{:.1f}".format(edge.id, edge.capacityUsed, edge.capacity))
            else: edgeFile.write("{:.0f}\t{:.0f}\t{:.0f}\t{:.1f}\t{:.1f}\n".format(edge.id, edge.u.id, edge.v.id, edge.capacityUsed, edge.capacity))
        routeFile.close()
        edgeFile.close()
    
    def dwDecomposition(self, M=1e6, epsilon=1e-6, output=0, outputFilefolder="./output/"):
        ## The list of extreme solutions: 极点的列表（对应DW算法中的列）
        self.solutions = []
        ## Find an initial extreme solution by assigning all the flow to the current shortest path: 使用最短路算法找到一个极点（对应一列）
        solution = self.generatePathForDemand()
        self.solutions.append(solution)
        ## Find edges with upper bounded capacity: 找到有容量上界的边
        self.boundedEdges = {}
        for edgeId, edge in self.edges.items():
            edge: Edge
            if edge.capacity < np.inf:
                self.boundedEdges[edgeId] = edge
        ## Initialization: 初始化模型
        masterProblem = gp.Model()
        masterProblem.Params.LogToConsole = 0
        masterProblem.Params.LogFile = "./log.txt"
        ## Add variables: 添加变量，包括lambda、松弛变量和人工变量
        lams = masterProblem.addVar(lb=0, ub=1, obj=self.solutions[0]["cost"], name="lam[0]") # lambda
        slacks = masterProblem.addVars(len(self.boundedEdges), lb=0, obj=0, name='s') # slack variables: 松弛变量
        surpluses = masterProblem.addVars(len(self.boundedEdges), lb=0, obj=M, name='a') # artifitial variables: 人工变量
        ## Add constraints (capacity constraints and sum(lambda) = 1): 添加约束，包括容量约束和sum(lambda)=1的约束
        masterProblem.addConstrs((self.solutions[0]["flow"][k] * lams + slacks[j] - surpluses[j] == self.boundedEdges[k].capacity for j, k in enumerate(self.boundedEdges.keys())), name="capacity")
        masterProblem.addConstr(lams == 1)
        ## Start! 记录开始时间
        startTime = time.time()
        ## Solve the Restricted Master Problem: 求解受限主问题RMP
        masterProblem.optimize()
        ## Get the dual: 得到对偶变量的值
        dualVars = masterProblem.getAttr(gp.GRB.Attr.Pi, masterProblem.getConstrs())
        ## Start the iteration: 迭代开始，迭代次数设置为0
        iterNum = 0
        ## Solve the Subproblem according to the dual, and find new column: 根据对偶变量的值求解子问题（SP），得到检验数，同时新的极点会添加到极点列表里
        reducedCost = self.subproblem(dualVars)
        ## Output: 输出
        iterationFile = open(outputFilefolder+"iterations.txt", 'w')
        iterationFile.write("iter_num\tobj\treduced_cost\n")
        if output: print("{:.0f}\t\t\t{:.2f}\t\t\t{:.2f}".format(iterNum, masterProblem.getObjective().getValue(), reducedCost))
        else: iterationFile.write("{:.0f}\t{:.2f}\t{:.2f}\n".format(iterNum, masterProblem.getObjective().getValue(), reducedCost))
        while reducedCost > epsilon and iterNum < 2000:
            ## Get the latest column: 取最新添加的极点
            s = self.solutions[-1]
            ## Calculate the coefficient of the new column: 计算该极点对应的新列的系数
            colCoeff = [s["flow"][k] for k in self.boundedEdges.keys()]
            colCoeff.append(1) # 别忘了lambda对应的系数1
            ## Add the new column to the model: 生成新列添加进模型
            column = gp.Column(colCoeff, masterProblem.getConstrs())
            masterProblem.addVar(lb=0, ub=1, obj=s["cost"], name="lam["+str(iterNum+1)+']', column=column)
            ## Solve the Restricted Master Problem: 求解受限主问题RMP
            masterProblem.optimize()
            ## Get the dual: 得到对偶变量的值
            dualVars = masterProblem.getAttr(gp.GRB.Attr.Pi, masterProblem.getConstrs())
            ## Solve the Subproblem according to the dual, and find new column: 根据对偶变量的值求解子问题（SP），得到检验数，同时新的极点会添加到极点列表里
            reducedCost = self.subproblem(dualVars)
            ## Output: 输出
            iterNum += 1
            if output: print("{:.0f}\t\t\t{:.2f}\t\t\t{:.2f}".format(iterNum, masterProblem.getObjective().getValue(), reducedCost))
            else: iterationFile.write("{:.0f}\t{:.2f}\t{:.2f}\n".format(iterNum, masterProblem.getObjective().getValue(), reducedCost))
        ## End! 记录结束时间
        endTime = time.time()
        ## Save the model: 模型保存和输出
        self.objModel = masterProblem
        iterationFile.close()
        print("Iteration time: {:.2f}s. Objective: {:.2f}.".format(endTime - startTime, masterProblem.getObjective().getValue()))
        varFile = open(outputFilefolder+"variables.txt", 'w')
        varFile.write("var_name\tvalue\n")
        for v in self.objModel.getVars():
            if v.X != 0:
                varName = v.VarName
                varValue = v.X
                if output: print(varName + '\t' + str(varValue))
                else: varFile.write("{:}\t{:.6f}".format(varName, varValue) + '\n')
        varFile.close()
        ## Process the final solution: 得到各OD的路径和流量信息
        self.retrieveSol(output, outputFilefolder)

if __name__ == "__main__":
    network = Network()
    network.loadNetwork()
    network.dwDecomposition()
    sensitivityFile = open("./output/sensitivity.txt", 'w')
    sensitivityFile.write("m\tobj\n")
    for m in range(15):
        network = Network()
        network.loadNetwork()
        network.dwDecomposition(m)
        sensitivityFile.write("{:.0f}\t{:.6f}\n".format(m, network.objModel.getObjective().getValue()))
    sensitivityFile.close()
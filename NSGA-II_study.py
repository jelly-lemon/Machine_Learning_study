"""
NSGA-II 遗传算法


"""


import math
import random
import matplotlib.pyplot as plt


def function1(x):
    """
    目标函数 1
    """
    value = -x ** 2
    return value


def function2(x):
    """
    目标函数 2

    我们的期望是在两个目标函数上都尽量取得比较大的值
    """
    value = -(x - 2) ** 2
    return value


# Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


def fast_non_dominated_sort(values1, values2):
    """
    快速非支配排序

    :param values1:
    :param values2:
    :return:
    """
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    # p，q 都表示可行解的编号
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            # 目标函数值大的解，我们认为是更好的解
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (
                    values1[p] >= values1[q] and values2[p] > values2[q]) or (
                    values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (
                    values1[q] >= values1[p] and values2[q] > values2[p]) or (
                    values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


def crowding_distance(values1, values2, front):
    """
    计算每个个体的拥挤度

    :param values1:
    :param values2:
    :param front:
    :return:
    """
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
    return distance


def crossover(a, b):
    """
    交叉

    :param a:
    :param b:
    :return:
    """
    r = random.random()
    if r > 0.5:
        return mutation((a + b) / 2)
    else:
        return mutation((a - b) / 2)


def mutation(solution):
    """
    变异

    :param solution:
    :return:
    """
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = min_x + (max_x - min_x) * random.random()
    return solution


# Main program starts here
pop_size = 20   # 种群大小
max_gen = 921   # 迭代（进化）次数

# Initialization
min_x = -55 # 个体最小值
max_x = 55  # 个体最大值
solution = [min_x + (max_x - min_x) * random.random() for i in range(0, pop_size)]  # 初始种群
gen_no = 0  # 当前迭代次数
while gen_no < max_gen:
    function1_values = [function1(solution[i]) for i in range(0, pop_size)]
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
    print("The best front for Generation number ", gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:
        print(round(solution[valuez], 3), end=" ")
    print("\n")
    crowding_distance_values = []
    for i in range(0, len(non_dominated_sorted_solution)):
        crowding_distance_values.append(
            crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]
    # Generating offsprings
    while len(solution2) != 2 * pop_size:
        a1 = random.randint(0, pop_size - 1)
        b1 = random.randint(0, pop_size - 1)
        solution2.append(crossover(solution[a1], solution[b1]))
    function1_values2 = [function1(solution2[i]) for i in range(0, 2 * pop_size)]
    function2_values2 = [function2(solution2[i]) for i in range(0, 2 * pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
    crowding_distance_values2 = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(
            crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
    new_solution = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [
            index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
            range(0, len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                 range(0, len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if (len(new_solution) == pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

# 绘制帕累托最优前沿
# 【注意】绘制帕累托最优前沿时，一般画左凹面
# 因此，如果直接画出来的结果是右凸面的话，可以对两个目标函数值加负号，
# 这样右凸面就变成了左凹面
# Q：加负号不就改变了目标函数值吗，这样还能是原来的意思吗？
# A：
# function1_values = [i * -1 for i in function1_values]
# function2_values = [j * -1 for j in function2_values]
fig, axes = plt.subplots(1, 3, figsize=(32, 9))
x = [i for i in range(-100, 100)]
y = [function1(r) for r in x]
axes[0].plot(x, y)
y = [function2(r) for r in x]
axes[1].plot(x, y)
axes[2].set_xlabel('Function 1', fontsize=15)
axes[2].set_ylabel('Function 2', fontsize=15)
axes[2].scatter(function1_values, function2_values)
plt.show()

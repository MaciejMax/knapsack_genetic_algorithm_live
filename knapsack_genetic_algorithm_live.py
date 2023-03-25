"""This code implements a genetic algorithm for solving the knapsack problem, 
which is a classic optimization problem in computer science. 
The goal is to maximize the total value of items that can be put into a knapsack 
with a limited weight capacity.

The code generates a set of items with random prices and weights, 
and then initializes a population of potential solutions (members), 
each represented by a binary genotype that indicates which items are included in the knapsack. 
The genetic algorithm then evolves the population over a series of generations, 
selecting the fittest members for breeding and mutation to create new offspring. 
The fitness of each member is determined by the total value of items in the knapsack, 
subject to the weight limit.

At the end of each generation, the code prints out the top 10 members (by fitness)
that were selected for breeding, and records their scores for plotting in a graph. 

The final output of the program is a graph of the scores of the top 10 members over time.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Final, List, Tuple

import matplotlib.pyplot as plt

# Constants - problem definition
_ITEMS_CNT_TO_GENERATE: Final[int] = 100
ITEMS: Final[List[Tuple[float, float]]] = [
    (price+1, math.log2(price+2))  # price, weight
    for price in range(_ITEMS_CNT_TO_GENERATE)
]
WEIGHT_LIMIT: Final[float] = 300

# Genetic algorithm parameters
# The best values for GENERATIONS_CNT and POPULATION_SIZE depend on various factors
# such as the complexity of the problem, the quality of the solutions required,
# the available computing resources, etc.
# In general, increasing the value of GENERATIONS_CNT can improve the quality
# of the solutions obtained, but it also increases the running time.
# Similarly, increasing the value of POPULATION_SIZE can lead to better diversity
# in the population, but it also increases the memory usage and computation time.
GENERATIONS_CNT: Final[int] = 10000
POPULATION_SIZE: Final[int] = 500
POPULATION_TOP_N_SIZE: Final[int] = POPULATION_SIZE // 2
assert POPULATION_TOP_N_SIZE < POPULATION_SIZE


@dataclass
class Member():
    """A dataclass to store genotype, score, and weight for a population member. 
    Members can be sorted by score."""
    genotype: list[0 | 1] = field(compare=False)
    score: float = 0.0
    weight: float = 0.0

    def __lt__(self, other):
        # ascending on score, descending on weight
        if self.score != other.score:
            return self.score < other.score
        return self.weight > other.weight


def get_random_population(population_size: int,
                          genotype_size: int) -> list[Member]:
    """Returns a list of randomly generated population members with a specified genotype size."""
    return [Member(genotype=[random.randint(0, 1) for _ in range(genotype_size)])
            for _ in range(population_size)]


def update_member_score_and_weight(member: Member):
    """Updates the score and weight for a population member 
    based on their genotype and the given items.
    Score is set to 0 if member exceeds weight limit, else total value"""
    assert len(member.genotype) == len(ITEMS)

    total_weight, total_value = 0.0, 0.0
    for idx, is_selected in enumerate(member.genotype):
        if is_selected == 1:
            total_value += ITEMS[idx][0]
            total_weight += ITEMS[idx][1]
    member.score = total_value if total_weight < WEIGHT_LIMIT else 0
    member.weight = total_weight


def mutate_member(population_member: Member):
    """Randomly mutates a population member's genotype (in-place)"""
    # mutate? 50% chance
    if random.randint(0, 1) == 0:
        genotype = population_member.genotype
        random_mutation_index = random.randint(0, len(genotype)-1)
        # flip 0-1
        genotype[random_mutation_index] = 1 - genotype[random_mutation_index]


def crossbreeding(x_member: Member,
                  y_member: Member) -> Member:
    """Creates a child member from two parent members through crossbreeding."""
    half_genotype_size = len(x_member.genotype)//2
    return Member(x_member.genotype[:half_genotype_size] + y_member.genotype[half_genotype_size:])


def main():
    """The main function to run the genetic algorithm. 
    Generates a random population, scores the population members, 
    selects the top performers for breeding, crossbreeds and mutates the offspring, 
    and replaces the old population with the new population. 
    Outputs information on the top-performing individuals in each generation 
    and graphs their scores over time."""

    # graph init
    graph_data = []
    # set interactive mode on
    plt.ion()
    _, axis = plt.subplots(nrows=1, ncols=2)

    # init
    population = get_random_population(population_size=POPULATION_SIZE,
                                       genotype_size=len(ITEMS))

    for generation_cnt in range(GENERATIONS_CNT):
        # SCORING
        for population_member in population:
            update_member_score_and_weight(population_member)

        # SELECTION FOR BREEDING"
        population.sort(reverse=True)
        members_selected_for_breeding = population[:POPULATION_TOP_N_SIZE]
        assert members_selected_for_breeding[0].score > 0, "Top 1 has score=0 - no hope, no future"

        # PRINT STATS
        top10 = members_selected_for_breeding[:10]
        print(f"\nGENERATION {generation_cnt}, ",
              f"top 10 out of {POPULATION_TOP_N_SIZE} selected for breeding: ")
        for population_member in top10:
            print(population_member)
        print("top 10 scores:", [int(population_member.score)
              for population_member in top10])

        # STORE GRAPH DATA
        # add 3395 reference line for the top score found so far
        graph_data.append([3395] +
                          [population_member.score for population_member in top10])

        # UPDATE PLOT
        if generation_cnt % 20 == 0:
            axis[0].clear()
            axis[0].set_title(f"Generations set:{GENERATIONS_CNT}")
            axis[0].plot(graph_data)
            axis[1].clear()
            axis[1].set_title("Last 100")
            axis[1].plot(graph_data[-100:])

            plt.draw()
            plt.pause(0.0001)

        # CREATE NEW POPULATION - CROSSBREEDING & MUTATION
        population = []
        # save the best one, without mutation ;)
        population.append(members_selected_for_breeding[0])
        # add the rest (-1)
        for _ in range(POPULATION_SIZE-1):
            newborn = crossbreeding(
                *random.sample(members_selected_for_breeding, 2))
            mutate_member(newborn)
            population.append(newborn)

    # SHOW GRAPH
    # turn interactive mode off and display the final plot
    plt.ioff()
    plt.show()
    print(members_selected_for_breeding[0])

if __name__ == "__main__":
    main()

"""tests for knapsack_genetic_algorithm"""
from knapsack_genetic_algorithm_live import Member, get_random_population, crossbreeding


def test_get_random_population():
    """test get_random_population"""
    population_size = 10
    genotype_size = 20
    population = get_random_population(population_size, genotype_size)
    assert len(population) == population_size
    assert all(len(member.genotype) == genotype_size for member in population)


def test_crossbreeding():
    """test crossbreeding"""
    x_genotype = [1, 0, 1, 1]
    y_genotype = [0, 1, 0, 1]
    x_member = Member(genotype=x_genotype)
    y_member = Member(genotype=y_genotype)
    child_member = crossbreeding(x_member, y_member)
    assert child_member.genotype == [1, 0, 0, 1]

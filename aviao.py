from random import Random
from time import time
from math import cos
from math import pi
from inspyred import ec
from inspyred.ec import terminators

from inspyred.ec.selectors import tournament_selection
from inspyred.ec.variators.crossovers import crossover

import numpy as np
import os


def generate_(random, args):
    size = args.get('num_inputs', 12)
    return [random.randint(0, 16000) for i in range(size)]


def evaluate_(candidates, args):
    fitness = []
    for cs in candidates:
        fit = perform_fitness(cs[0], cs[1], cs[2], cs[3], cs[4],
                              cs[5], cs[6], cs[7], cs[8], cs[9], cs[10], cs[11])
        fitness.append(fit)
    return fitness


def perform_fitness(c1_d, c1_c, c1_t, c2_d, c2_c, c2_t, c3_d, c3_c, c3_t, c4_d, c4_c, c4_t):

    # variaveis da carga1 de cada compartimento
    c1_d = np.round(c1_d)
    c1_c = np.round(c1_c)
    c1_t = np.round(c1_t)

    # variaveis da carga2 de cada compartimento
    c2_d = np.round(c2_d)
    c2_c = np.round(c2_c)
    c2_t = np.round(c2_t)

    # variaveis da carga3 de cada compartimento
    c3_d = np.round(c3_d)
    c3_c = np.round(c3_c)
    c3_t = np.round(c3_t)

    # variaveis da carga4 de cada compartimento
    c4_d = np.round(c4_d)
    c4_c = np.round(c4_c)
    c4_t = np.round(c4_t)

    fit = float((0.31 * (c1_d + c1_c + c1_t) + 0.38 * (c2_d + c2_c + c2_t) + 
                0.35 * (c3_d + c3_c + c3_t) + 0.285 * (c4_d + c4_c + c4_t)) / 22750)

    # Satisfação das restrições em kg das cargas
    h1 = np.maximum(0, float((c1_d + c1_c + c1_t) - 18000)) / (18000 / 13)
    h2 = np.maximum(0, float((c2_d + c2_c + c2_t) - 15000)) / (15000 / 13)
    h3 = np.maximum(0, float((c3_d + c3_c + c3_t) - 23000)) / (23000 / 13)
    h4 = np.maximum(0, float((c4_d + c4_c + c4_t) - 12000)) / (12000 / 13)

    # Satisfação das restrições em kg dos compartimentos
    h5 = np.maximum(0, float((c1_d + c2_d + c3_d + c4_d) - 10000)) / (10000 / 13)
    h6 = np.maximum(0, float((c1_c + c2_c + c3_c + c4_c) - 16000)) / (16000 / 13)
    h7 = np.maximum(0, float((c1_t + c2_t + c3_t + c4_t) - 8000)) / (8000 / 13)

    # Satisfação das restrições dos volumes dos compartimentos
    h8 = np.maximum(0, float((c1_d * 0.48 + c2_d * 0.65 + c3_d * 0.58 + c4_d * 0.39) - 6800)) / (6800 / 13)
    h9 = np.maximum(0, float((c1_c * 0.48 + c2_c * 0.65 + c3_c * 0.58 + c4_c * 0.39) - 8700)) / (8700 / 13)
    h10 = np.maximum(0, float((c1_t * 0.48 + c2_t * 0.65 +c3_t * 0.58 + c4_t * 0.39) - 5300)) / (5300 / 13)

    # Satisfação das proporções dos compartimentos
    carga_total = float(c1_d + c2_d + c3_d + c4_d + c1_c + c2_c + c3_c + c4_c + c1_t + c2_t + c3_t + c4_t)
    h11 = np.maximum(0, float((((c1_d + c2_d + c3_d + c4_d) / carga_total) - (10000 / 34000)) / ((10000 / 34000) / 13)))
    h12 = np.maximum(0, float((((c1_c + c2_c + c3_c + c4_c) / carga_total) - (16000 / 34000)) / (16000 / 34000) / 13))
    h13 = np.maximum(0, float((((c1_t + c2_t + c3_t + c4_t) / carga_total) - (8000 / 34000)) / ((8000 / 34000) / 13)))

    fit = fit - (h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9 + h10 + h11 + h12 + h13)

    return fit


def solution_evaluation(c1_d, c1_c, c1_t, c2_d, c2_c, c2_t, c3_d, c3_c, c3_t, c4_d, c4_c, c4_t):

    c1_d = np.round(c1_d)
    c1_c = np.round(c1_c)
    c1_t = np.round(c1_t)
    c2_d = np.round(c2_d)
    c2_c = np.round(c2_c)
    c2_t = np.round(c2_t)
    c3_d = np.round(c3_d)
    c3_c = np.round(c3_c)
    c3_t = np.round(c3_t)
    c4_d = np.round(c4_d)
    c4_c = np.round(c4_c)
    c4_t = np.round(c4_t)

    print("..: RESUMO DA ORGANIZACAO: ..")
    print("Lucro total: ", float(0.31 * (c1_d + c1_c + c1_t) + 0.38 * (c2_d +
          c2_c + c2_t) + 0.35 * (c3_d + c3_c + c3_t) + 0.285 * (c4_d + c4_c + c4_t)))
    print("")
    print("Carga 1 dianteiro: ", c1_d)
    print("Carga 1 central: ", c1_c)
    print("Carga 1 traseiro: ", c1_t)
    print("Carga 1 total: ", c1_d + c1_c + c1_t)
    print("")
    print("Carga 2 dianteiro: ", c2_d)
    print("Carga 2 central: ", c2_c)
    print("Carga 2 traseiro: ", c2_t)
    print("Carga 2 total: ", c2_d + c2_c + c2_t)
    print("")
    print("Carga 3 dianteiro: ", c3_d)
    print("Carga 3 central: ", c3_c)
    print("Carga 3 traseiro: ", c3_t)
    print("Carga 3 total: ", c3_d + c3_c + c3_t)
    print("")
    print("Carga 4 dianteiro: ", c4_d)
    print("Carga 4 central: ", c4_c)
    print("Carga 4 traseiro: ", c4_t)
    print("Carga 4 total: ", c4_d + c4_c + c4_t)
    print("")
    print("Carga total no compartimento dianteiro:", c1_d + c2_d + c3_d + c4_d)
    print("Carga total no compartimento cenral:", c1_c + c2_c + c3_c + c4_c)
    print("Carga total no compartimento traseiro:", c1_t + c2_t + c3_t + c4_t)
    print("")
    print("Volume ocupado do compartimento dianteiro: ", float((c1_d * 0.48) + (c2_d * 0.65) + (c3_d * 0.58) + (c4_d * 0.39)))
    print("Volume ocupado do compartimento central: ", float((c1_c * 0.48) + (c2_c * 0.65) + (c3_c * 0.58) + (c4_c * 0.39)))
    print("Volume ocupado do compartimento traseiro: ", float((c1_t * 0.48) + (c2_t * 0.65) + (c3_t * 0.58) + (c4_t * 0.39)))
    print("")
    carga_total = float(c1_d + c2_d + c3_d + c4_d + c1_c + c2_c + c3_c + c4_c + c1_t + c2_t + c3_t + c4_t)
    print("Proporcao dianteiro: ", float((c1_d + c2_d + c3_d + c4_d) / carga_total))
    print("Proporcao central: ", float((c1_c + c2_c + c3_c + c4_c) / carga_total))
    print("Proporcao traseiro: ", float((c1_t + c2_t + c3_t + c4_t) / carga_total))


def main():
    rand = Random()
    rand.seed(int(time()))

    ea = ec.GA(rand)
    ea.selector = ec.selectors.tournament_selection
    ea.variator = [ec.variators.uniform_crossover,
                   ec.variators.gaussian_mutation]
    ea.replacer = ec.replacers.steady_state_replacement

    ea.terminator = terminators.generation_termination

    ea.observer = [ec.observers.stats_observer, ec.observers.file_observer]

    final_pop = ea.evolve(
        generator=generate_,
        evaluator=evaluate_,
        pop_size=1000,
        maximize=True,
        bounder=ec.Bounder(0, 16000),
        max_generations=10000,
        num_inputs=12,
        crossover_rate=1.0,
        num_crossover_points=1,
        mutation_rate=0.25,
        num_elites=1,
        num_selected=12,
        tournament_size=12,
        statistics_file=open("cargas_stats.csv", "w"),
        individuals_file=open("cargas_individuals.csv", "w"))

    final_pop.sort(reverse=True)
    print(final_pop[0])

    perform_fitness(final_pop[0].candidate[0], final_pop[0].candidate[1], final_pop[0].candidate[2], final_pop[0].candidate[3], final_pop[0].candidate[4], final_pop[0].candidate[5], final_pop[0].candidate[6], final_pop[0].candidate[7],
                    final_pop[0].candidate[8], final_pop[0].candidate[9], final_pop[0].candidate[10], final_pop[0].candidate[11])
    solution_evaluation(final_pop[0].candidate[0], final_pop[0].candidate[1], final_pop[0].candidate[2], final_pop[0].candidate[3], final_pop[0].candidate[4], final_pop[0].candidate[5], final_pop[0].candidate[6], final_pop[0].candidate[7],
                        final_pop[0].candidate[8], final_pop[0].candidate[9], final_pop[0].candidate[10], final_pop[0].candidate[11])


main()
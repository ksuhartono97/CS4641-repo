package com.company;


import java.util.Arrays;
import java.util.stream.DoubleStream;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

public class FourPeaks {
    /** The n value */
    private static final int N = 100;
    /** The t value */
    private static final int T = N / 10;

    private static int numberOfLoops = 20;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double[] scoreArray = new double[numberOfLoops];

        FixedIterationTrainer fit;

        for (int i = 0; i < numberOfLoops; i++) {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            scoreArray[i] = ef.value(rhc.getOptimal());
        }

        System.out.println("RHC: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);


        scoreArray = new double[numberOfLoops];
        for (int i = 0; i < numberOfLoops; i++) {
            SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            scoreArray[i] = ef.value(sa.getOptimal());
        }

        System.out.println("SA: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);

        scoreArray = new double[numberOfLoops];
        for (int i = 0; i < numberOfLoops; i++) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            scoreArray[i] = ef.value(ga.getOptimal());
        }

        System.out.println("GA: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);


        scoreArray = new double[numberOfLoops];
        for (int i = 0; i < numberOfLoops; i++) {
            MIMIC mimic = new MIMIC(200, 20, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            scoreArray[i] = ef.value(mimic.getOptimal());
        }
        System.out.println("MIMIC: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
    }
}

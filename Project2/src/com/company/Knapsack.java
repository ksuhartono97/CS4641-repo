package com.company;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;
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
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knapsack problem
 *
 * Given a set of items, each with a weight and a value, determine the number of each item to include in a
 * collection so that the total weight is less than or equal to a given limit and the total value is as
 * large as possible.
 * https://en.wikipedia.org/wiki/Knapsack_problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class Knapsack {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 50;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum value for a single element */
    private static final double MAX_VALUE = 50;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum weight for the knapsack */
    private static final double MAX_KNAPSACK_WEIGHT =
            MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .25;
    private static DecimalFormat deci = new DecimalFormat("0.000");
    private static int numberOfLoops = 10;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        EvaluationFunction ef = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);

        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double starttime = 0;
        double endtime = 0;
        double time_elapsed = endtime - starttime;

        double[] timeArray = new double[numberOfLoops];
        double[] scoreArray = new double[numberOfLoops];

        FixedIterationTrainer fit;

        for (int i = 0; i < numberOfLoops; i++) {
            starttime = System.nanoTime();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();

            endtime = System.nanoTime();
            time_elapsed = endtime - starttime;
            time_elapsed /= Math.pow(10, 9);

            timeArray[i] = time_elapsed;
            scoreArray[i] = ef.value(rhc.getOptimal());
        }
        System.out.println("RHC: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
        System.out.printf("Time Elapsed: %s s %n", deci.format( DoubleStream.of(timeArray).sum() / timeArray.length));

        timeArray = new double[numberOfLoops];
        scoreArray = new double[numberOfLoops];

        for (int i = 0; i < numberOfLoops; i++) {
            starttime = System.nanoTime();
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            endtime = System.nanoTime();
            time_elapsed = endtime - starttime;
            time_elapsed /= Math.pow(10, 9);

            timeArray[i] = time_elapsed;
            scoreArray[i] = ef.value(sa.getOptimal());
        }

        System.out.println("SA: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
        System.out.printf("Time Elapsed: %s s %n", deci.format( DoubleStream.of(timeArray).sum() / timeArray.length));

        timeArray = new double[numberOfLoops];
        scoreArray = new double[numberOfLoops];

        for (int i = 0; i < numberOfLoops; i++) {
            starttime = System.nanoTime();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            endtime = System.nanoTime();
            time_elapsed = endtime - starttime;
            time_elapsed /= Math.pow(10, 9);

            timeArray[i] = time_elapsed;
            scoreArray[i] = ef.value(ga.getOptimal());
        }

        System.out.println("GA: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
        System.out.printf("Time Elapsed: %s s %n", deci.format( DoubleStream.of(timeArray).sum() / timeArray.length));


        timeArray = new double[numberOfLoops];
        scoreArray = new double[numberOfLoops];

        for (int i = 0; i < numberOfLoops; i++) {
            starttime = System.nanoTime();
            MIMIC mimic = new MIMIC(200, 100, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            endtime = System.nanoTime();
            time_elapsed = endtime - starttime;
            time_elapsed /= Math.pow(10, 9);

            timeArray[i] = time_elapsed;
            scoreArray[i] = ef.value(mimic.getOptimal());
        }

        System.out.println("MIMIC: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
        System.out.printf("Time Elapsed: %s s %n", deci.format( DoubleStream.of(timeArray).sum() / timeArray.length));
    }

}
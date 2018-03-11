package com.company;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.DoubleStream;

import opt.ga.NQueensFitnessFunction;
import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author kmanda1
 * @version 1.0
 */
public class NQueensTest {
    /** The n value */
    private static final int N = 140;
    /** The t value */

    private static DecimalFormat deci = new DecimalFormat("0.000");
    private static int numberOfLoops = 10;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Random random = new Random(N);
        for (int i = 0; i < N; i++) {
            ranges[i] = random.nextInt();
        }
        NQueensFitnessFunction ef = new NQueensFitnessFunction();
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double[] timeArray = new double[numberOfLoops];
        double[] scoreArray = new double[numberOfLoops];

        FixedIterationTrainer fit;
        long starttime = 0;

        for (int i = 0; i < numberOfLoops; i++) {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            fit = new FixedIterationTrainer(rhc, 100);
            fit.train();
            starttime = System.currentTimeMillis();
            scoreArray[i] = ef.value(rhc.getOptimal());
            timeArray[i] = (System.currentTimeMillis() - starttime);
        }

        System.out.println("RHC: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
        System.out.printf("Time Elapsed: %s ms %n", deci.format( DoubleStream.of(timeArray).sum() / timeArray.length));
//        System.out.println("RHC: Board Position: ");
        // System.out.println(ef.boardPositions());

        System.out.println("============================");

        timeArray = new double[numberOfLoops];
        scoreArray = new double[numberOfLoops];

        for (int i = 0; i < numberOfLoops; i++) {
            SimulatedAnnealing sa = new SimulatedAnnealing(1E1, .1, hcp);
            fit = new FixedIterationTrainer(sa, 100);
            fit.train();
            starttime = System.currentTimeMillis();
            scoreArray[i] = ef.value(sa.getOptimal());
            timeArray[i] = (System.currentTimeMillis() - starttime);
        }

        System.out.println("SA: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
        System.out.printf("Time Elapsed: %s ms %n", deci.format( DoubleStream.of(timeArray).sum() / timeArray.length));
//        System.out.println("SA: Board Position: ");
        // System.out.println(ef.boardPositions());

        System.out.println("============================");

        timeArray = new double[numberOfLoops];
        scoreArray = new double[numberOfLoops];

        for (int i = 0; i < numberOfLoops; i++) {
            starttime = System.currentTimeMillis();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 0, 10, gap);
            fit = new FixedIterationTrainer(ga, 100);
            fit.train();
            starttime = System.currentTimeMillis();
            scoreArray[i] = ef.value(ga.getOptimal());
            timeArray[i] = (System.currentTimeMillis() - starttime);
        }
        System.out.println("GA: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
        System.out.printf("Time Elapsed: %s ms %n", deci.format( DoubleStream.of(timeArray).sum() / timeArray.length));
//        System.out.println("GA: Board Position: ");
        //System.out.println(ef.boardPositions());

        System.out.println("============================");

        timeArray = new double[numberOfLoops];
        scoreArray = new double[numberOfLoops];

        for (int i = 0; i < numberOfLoops; i++) {
            starttime = System.currentTimeMillis();
            MIMIC mimic = new MIMIC(200, 10, pop);
            fit = new FixedIterationTrainer(mimic, 5);
            fit.train();
            starttime = System.currentTimeMillis();
            scoreArray[i] = ef.value(mimic.getOptimal());
            timeArray[i] = (System.currentTimeMillis() - starttime);
        }
        System.out.println("MIMIC: " + DoubleStream.of(scoreArray).sum() / scoreArray.length);
        System.out.printf("Time Elapsed: %s ms %n", deci.format( DoubleStream.of(timeArray).sum() / timeArray.length));
//        System.out.println("MIMIC: Board Position: ");
        //System.out.println(ef.boardPositions());
    }
}

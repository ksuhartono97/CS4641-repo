package com.company;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import util.linalg.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.*;

public class Poker2 {

    private static final String FILENAME = "data/train.csv";
    private static int num_examples = 25010;
    private static int num_attributes = 10;


  /* Randomization */
    /**
     * Randomizing your rows
     * <p>
     * If you enable the randomization, your rows will be shuffled.
     * The seed value can be any arbitrary value. Having a seed ensures that each run of the
     * script has the same randomized shuffle ordering as any other run.
     */
    private static final boolean shouldRandomize = true;
    private static final long SEED = 0xABCDEF;

  /* Cross Validation and Testing params */
    /**
     * Set K for number of folds.
     * Set PERCENT_TRAIN for the percentage to be used for training
     */
    private static final int K = 5;
    private static final double PERCENT_TRAIN = 0.8;


  /* Neural Network Params*/
    /**
     * Tweak the following as needed. The maxLearningRate and minLearningRate are the default
     * hardcoded values in RPROPUpdateRule.java
     * <p>
     * The accuracy threshold to stop backprop. Set this EXTREMELY low to ensure training ends only
     * when absolutely sure network output is correct
     */
    private static double initialLearningRate = 0.1;
    private static double maxLearningRate = 50;
    private static double minLearningRate = 0.000001;
    private static double backprop_threshold = 1e-10;

  /* Backprop Params */
    /**
     * If true, shouldFindNNParams determines the best params for your neural network. Since this
     * process is lengthy, you don't want to repeat it often. Do it once, record the value, and
     * place the value for numberHiddenLayerNodes.
     */
    private static final boolean shouldFindNNParams = false;
    private static int numberHiddenLayerNodes = 50;


  /* Simulated Annealing */
    /**
     * Same as above, if you've already run it once, just record the values and store them in
     * best_temp and best_cooling. Otherwise, the temps[] and cooling[] values will be cycled
     * through until the best param configuration for simulated annealing is found.
     */
//  TODO: Set this to false if you retained the best SA params from a previous run
    private static final boolean shouldFindSAParams = false;
    // TODO: Modify these to try different possible temp and cooling params
    private static double[] temps = {100};
    private static double[] coolingRates = {0.5, 0.52, 0.54, 0.56, 0.58, 0.6};

    //  TODO: Place values here from previous run if you set shouldFindSAParams to false.
    private static double best_temp = 100;
    private static double best_cooling = 0.6;


  /* Genetic Algorithms */
    /**
     * Same as above, if you've already run it once, just record the values and store them in
     * populationSize, toMate, and toMutate. Otherwise, the ratios will be cycled
     * through until the best param configuration for genetic algorithms is found.
     * <p>
     * NOTE: min(populationRations) >= max(mateRatios) + max(toMutateRatio)
     * This condition must be upheld or Exceptions will be thrown later in the script
     */
    private static final boolean shouldFindGAParams = true;
    private static double[] populationRatios = {0.15, 0.20, 0.25, 0.30};
    private static double[] mateRatios = {0.02, 0.04, 0.06};
    private static double[] mutateRatios = {0.02, 0.04, 0.06};

    //  TODO: Place values here from previous run if you set shouldFindGAParams to false
    private static double populationRatio = 0.25;
    private static double toMateRatio = 0.02;
    private static double toMutateRatio = 0.02;


  /* Other global vars - Don't mess with these unless you have a reason to */

    //  Number of input nodes is the same as the number of attributes
    private static int inputLayer = num_attributes;
    //  This is determined dynamically
    private static int outputLayer;

    //  Determines how many possible classifications we can have
    private static Set<String> unique_labels;

    private static Instance[] allInstances;
    private static Map<String, double[]> bitvector_mappings;

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DecimalFormat df = new DecimalFormat("0.000");

    //  Train and Test sets
    private static DataSet trainSet;
    private static DataSet testSet;
    private static DataSet[] splittedTrainSet;
    //  Cross validation folds
    private static List<Instance[]> folds;


  /* Begin actual code */

    /**
     * Comment out anything you don't want to immediately run.
     * Advice: Write loops to do certain numbers of iterations and output results for rand opt
     * algorithms
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        initializeInstances();

        /* Actual Train Test Code */


        int trainIterations;
        makeTestTrainSets();

        for (int i = 0; i < 5; i++) {
            trainSet = splittedTrainSet[i];
            folds = kfolds(splittedTrainSet[i]);

            System.out.println("===========");
            System.out.println("Current sample size: " + trainSet.size());
            System.out.println("===========");

            //Back Propagation
//            trainIterations = 2000;
//            runBackprop(trainIterations);

            //RHC
//            trainIterations = 1200;
//            runRHC(trainIterations);

            //SA
//            trainIterations = 4000;
//            runSA(trainIterations, best_temp, best_cooling);

            //GA
            trainIterations = 10;
            runGA(trainIterations);
        }


        /* Param Tuning Code */

        makeTestTrainSets();
        folds = kfolds(trainSet);


//    Determine best NN params, which happen to just be the number of hidden layer nodes.
//        if (shouldFindNNParams) {
//
////      This is how Weka does it: # of attributes + # of classes / 2
//            int weka_technique = num_attributes + outputLayer / 2;
//            int[] hiddenLayerNodes = {weka_technique, 5, 10, 20, 50, 100, 200};
//            trainIterations = 200;
//            determineNNParams(trainIterations, hiddenLayerNodes);
//        }

    /* Backprop */
//    for(int i = 500; i < 2100; i+=500) {
//        trainIterations = i;
//        runBackprop(trainIterations);
//    }

    /* RHC */

//    for (int i = 0; i < 1600; i+=100) {
//        trainIterations = i;
//        runRHC(trainIterations);
//    }

    /* SA */
//        if (shouldFindSAParams) {
//            trainIterations = 100;
//            determineSAParams(trainIterations);
//        }

//        for (int i = 500; i < 4100; i+=500) {
//            System.out.print("== Iterations: " );
//            System.out.print(i);
//            System.out.println(" ==");
//            trainIterations = i;
//            runSA(trainIterations, best_temp, best_cooling);
//        }


//
    /* GA */
        if (shouldFindGAParams) {
////      TODO: Keep this very small
            trainIterations = 10;
            determineGAParams(trainIterations);
//
        }
//

    }


    /**
     * Uses k-folds cross validation to determine the best number of hidden layers to configure a
     * nueral network among the list provided
     *
     * @param trainIterations  the number of iterations to run
     * @param hiddenLayerNodes an int[] of hidden layer nodes to try
     */
    public static void determineNNParams(int trainIterations, int[] hiddenLayerNodes) {

        System.out.println("===========Cross Validation for NN Params=========");
        double[] errors = new double[hiddenLayerNodes.length];
        for (int m = 0; m < hiddenLayerNodes.length; m++) {

            int numHiddenNodes = hiddenLayerNodes[m];

            BackPropagationNetwork[] nets = new BackPropagationNetwork[K];
            double[] validationErrors = new double[K];

//      Create and train networks
            for (int i = 0; i < nets.length; i++) {

                Instance[] validation = getValidationFold(folds, i);
                Instance[] trainFolds = getTrainFolds(folds, i);
                DataSet trnfoldsSet = new DataSet(trainFolds);

//        Creates a network with specified number of hidden layer nodes
                nets[i] = factory.createClassificationNetwork(
                        new int[]{inputLayer, numHiddenNodes, outputLayer});

                BackPropagationNetwork backpropNet = nets[i];

                ConvergenceTrainer trainer = new ConvergenceTrainer(
                        new BatchBackPropagationTrainer(trnfoldsSet, backpropNet, new SumOfSquaresError(),
                                new RPROPUpdateRule(initialLearningRate, maxLearningRate, minLearningRate)),
                        backprop_threshold, trainIterations);

                trainer.train();

                validationErrors[i] = evaluateNetwork(backpropNet, validation);
            }

//      Find the average error for this network configuration
            double error = 0;
            for (int j = 0; j < validationErrors.length; j++) {
                error += validationErrors[j];
            }
            error /= validationErrors.length;
            errors[m] = error;

            System.out.printf("Nodes: %d\tError: %s%%%n", numHiddenNodes, df.format(errors[m]));
        }

//    Find the index with the min avg validation error
        int best_index = 0;
        double minError = Double.MAX_VALUE;
        for (int j = 0; j < errors.length; j++) {
            if (errors[j] < minError) {
                minError = errors[j];
                best_index = j;
            }
        }
        int bestNumNodes = hiddenLayerNodes[best_index];

        System.out.printf("%nBest Num Hidden Nodes: %d\tError: %s%%%n", bestNumNodes, df.format
                (minError));

        numberHiddenLayerNodes = bestNumNodes;
    }


    /**
     * This method will run Backpropagation using each
     * combination of (K-1) folds for training, and the Kth fold for validation. Once the model
     * with the lowest validation set error is found, that is used as the "best" model and the
     * training and test errors on that model are recorded.
     *
     * @param trainIterations the number of epochs/iterations
     */
    public static void runBackprop(int trainIterations) {

        System.out.println("===========Backpropagation=========");

        double starttime = System.nanoTime();
        ;
        double endtime;

        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

        ConvergenceTrainer trainer = new ConvergenceTrainer(
                new BatchBackPropagationTrainer(trainSet, backpropNet, new SumOfSquaresError(),
                        new RPROPUpdateRule(initialLearningRate, maxLearningRate, minLearningRate)),
                backprop_threshold, trainIterations);

        trainer.train();

        double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
        double testError = evaluateNetwork(backpropNet, testSet.getInstances());

        System.out.printf("Train Error: %s%% %n", df.format(trainError));
        System.out.printf("Test Error: %s%% %n", df.format(testError));

        endtime = System.nanoTime();
        double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
        time_elapsed /= Math.pow(10, 9);
        System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));

    }


    /**
     * Determines optimal weights for configured neural network using Randomized Hill Climbing with
     * Random Restarts and evaluates a neural networks performance on train and test sets with
     * those weights
     *
     * @param trainIterations the number of iterations
     */
    public static void runRHC(int trainIterations) {

        System.out.println("===========Randomized Hill Climbing=========");

        double starttime = System.nanoTime();
        ;
        double endtime;

        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
                backpropNet, measure);

        OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);

//      TODO: Vary the number of iterations as needed for your results
        train(oa, backpropNet, trainIterations);

        double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
        double testError = evaluateNetwork(backpropNet, testSet.getInstances());

        System.out.printf("Train Error: %s%% %n", df.format(trainError));
        System.out.printf("Test Error: %s%% %n", df.format(testError));

        endtime = System.nanoTime();
        double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
        time_elapsed /= Math.pow(10, 9);
        System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));

    }


    public static void determineSAParams(int trainIterations) {

        System.out.println("===========Determining Simulated Annealing Params=========");

        double[][] errors = new double[temps.length][coolingRates.length];

        for (int x = 0; x < temps.length; x++) {

            double temp = temps[x];

            for (int y = 0; y < coolingRates.length; y++) {

                double cooling = coolingRates[y];

                BackPropagationNetwork[] nets = new BackPropagationNetwork[K];
                NeuralNetworkOptimizationProblem[] nnops = new NeuralNetworkOptimizationProblem[K];
                OptimizationAlgorithm[] oas = new OptimizationAlgorithm[K];
                double[] validationErrors = new double[K];

                for (int i = 0; i < nets.length; i++) {

                    Instance[] validation = getValidationFold(folds, i);
                    Instance[] trainFolds = getTrainFolds(folds, i);
                    DataSet trnfoldsSet = new DataSet(trainFolds);

//        Creates a network with specified number of hidden layer nodes
                    nets[i] = factory.createClassificationNetwork(
                            new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

                    BackPropagationNetwork backpropNet = nets[i];

                    nnops[i] = new NeuralNetworkOptimizationProblem(trnfoldsSet, backpropNet, measure);
                    oas[i] = new SimulatedAnnealing(temp, cooling, nnops[i]);

                    train(oas[i], nets[i], trainIterations);

                    validationErrors[i] = evaluateNetwork(backpropNet, validation);
                }

//      Find the average error for this network configuration
                double error = 0;
                for (int j = 0; j < validationErrors.length; j++) {
                    error += validationErrors[j];
                }
                error /= validationErrors.length;
                errors[x][y] = error;

                System.out.println("=====");
                System.out.println("Temp Index: " + temps[x]);
                System.out.println("Cooling Index: " + coolingRates[y]);
                System.out.println("Error: " + error);

            }
        }

        int best_temp_index = 0;
        int best_cool_index = 0;
        double minErr = Double.MAX_VALUE;

        for (int x = 0; x < temps.length; x++) {

            for (int y = 0; y < coolingRates.length; y++) {

                if (minErr > errors[x][y]) {
                    best_temp_index = x;
                    best_cool_index = y;
                    minErr = errors[x][y];
                }
            }
        }

        double bestCooling = coolingRates[best_cool_index];
        double bestTemp = temps[best_temp_index];
        System.out.printf("Best Cooling: %s%n", df.format(bestCooling));
        System.out.printf("Best Temp: %s%n", df.format(bestTemp));

    }


    public static void runSA(int trainIterations, double temp, double cooling) {

        System.out.println("===========Simulated Annealing=========");

        double starttime = System.nanoTime();
        ;
        double endtime;

        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
                backpropNet, measure);

        OptimizationAlgorithm oa = new SimulatedAnnealing(temp, cooling, nnop);

        train(oa, backpropNet, trainIterations);


        double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
        double testError = evaluateNetwork(backpropNet, testSet.getInstances());

        System.out.printf("Train Error: %s%% %n", df.format(trainError));
        System.out.printf("Test Error: %s%% %n", df.format(testError));

        endtime = System.nanoTime();
        double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
        time_elapsed /= Math.pow(10, 9);
        System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));
    }


    public static void determineGAParams(int trainIterations) {

        System.out.println("===========Determining Genetic Algorithms Params=========");

        double[][][] errors =
                new double[populationRatios.length][mateRatios.length][mutateRatios.length];

//    Training population size is always 9/10 of the total training set, or equivalently
//    9 times the validation set
        int trainingPopulation = getValidationFold(folds, 0).length * 9;

        for (int x = 0; x < populationRatios.length; x++) {

            int population = (int) (Math.max(populationRatios[x] * trainingPopulation, 10));

            for (int y = 0; y < mateRatios.length; y++) {

                int mate = (int) (Math.max(mateRatios[y] * trainingPopulation, 10));

                for (int z = 0; z < mutateRatios.length; z++) {

                    int mutate = (int) (Math.max(mutateRatios[z] * trainingPopulation, 10));

                    BackPropagationNetwork[] nets = new BackPropagationNetwork[K];
                    NeuralNetworkOptimizationProblem[] nnops = new NeuralNetworkOptimizationProblem[K];
                    OptimizationAlgorithm[] oas = new OptimizationAlgorithm[K];
                    double[] validationErrors = new double[K];

                    for (int i = 0; i < nets.length; i++) {

                        Instance[] validation = getValidationFold(folds, i);
                        Instance[] trainFolds = getTrainFolds(folds, i);
                        DataSet trnfoldsSet = new DataSet(trainFolds);

                        nets[i] = factory.createClassificationNetwork(
                                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

                        BackPropagationNetwork backpropNet = nets[i];

                        nnops[i] = new NeuralNetworkOptimizationProblem(trnfoldsSet, backpropNet, measure);
                        oas[i] = new StandardGeneticAlgorithm(population, mate, mutate, nnops[i]);

                        train(oas[i], backpropNet, trainIterations);

                        validationErrors[i] = evaluateNetwork(backpropNet, validation);
                    }

//      Find the average error for this configuration
                    double error = 0;
                    for (int j = 0; j < validationErrors.length; j++) {
                        error += validationErrors[j];
                    }
                    error /= validationErrors.length;
                    errors[x][y][z] = error;

                    System.out.println("=====");
                    System.out.println("Population Ratio: " + populationRatios[x]);
                    System.out.println("Mate Ratio: " + mateRatios[y]);
                    System.out.println("Mutate Ratio: " + mutateRatios[z]);
                    System.out.println("Error: " + error);

                }

            }

        }

        int best_pop = 0;
        int best_mate = 0;
        int best_mutate = 0;
        double minErr = Double.MAX_VALUE;

        for (int x = 0; x < populationRatios.length; x++) {

            for (int y = 0; y < mateRatios.length; y++) {

                for (int z = 0; z < mutateRatios.length; z++) {

                    if (errors[x][y][z] < minErr) {
                        best_pop = x;
                        best_mate = y;
                        best_mutate = z;
                        minErr = errors[x][y][z];
                    }
                }
            }
        }

        populationRatio = populationRatios[best_pop];
        toMateRatio = mateRatios[best_mate];
        toMutateRatio = mutateRatios[best_mutate];

        System.out.printf("Population Ratio: %s%n", df.format(populationRatio));
        System.out.printf("Mate Ratio: %s%n", df.format(toMateRatio));
        System.out.printf("Mutate Ratio: %s%n", df.format(toMutateRatio));
        System.out.printf("Error: %s%% %n", df.format(minErr));

    }

    /**
     * Run genetic algorithms.
     *
     * @param trainIterations the iterations to run
     */
    public static void runGA(int trainIterations) {

        System.out.println("===========Genetic Algorithms=========");

        double starttime = System.nanoTime();
        double endtime;

        int trainSetSize = trainSet.size();
        int populationSize = (int) (trainSetSize * populationRatio);
        int toMate = (int) (trainSetSize * toMateRatio);
        int toMutate = (int) (trainSetSize * toMutateRatio);

        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
                backpropNet, measure);

        OptimizationAlgorithm oa = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nnop);

        train(oa, backpropNet, trainIterations);


        double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
        double testError = evaluateNetwork(backpropNet, testSet.getInstances());

        System.out.printf("Train Error: %s%% %n", df.format(trainError));
        System.out.printf("Test Error: %s%% %n", df.format(testError));

        endtime = System.nanoTime();
        double time_elapsed = endtime - starttime;

//    Convert nanoseconds to seconds
        time_elapsed /= Math.pow(10, 9);
        System.out.printf("Time Elapsed: %s s %n", df.format(time_elapsed));
    }


    /**
     * Train a given optimization problem for a given number of iterations. Called by RHC, SA, and
     * GA algorithms
     *
     * @param oa         the optimization algorithm
     * @param network    the network that corresponds to the randomized optimization problem. The
     *                   optimization algorithm will determine the best weights to try using with this
     *                   network and assign those weights
     * @param iterations the number of training iterations
     */
    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, int
            iterations) {

        for (int i = 0; i < iterations; i++) {
            oa.train();
        }
        Instance optimalWeights = oa.getOptimal();
        network.setWeights(optimalWeights.getData());
    }


    /**
     * Given a network and instances, the output of the network is evaluated and a decimal value
     * for error is given
     *
     * @param network the BackPropagationNetwork with weights already initialized
     * @param data    the instances to be evaluated against
     * @return
     */
    public static double evaluateNetwork(BackPropagationNetwork network, Instance[] data) {

        double num_incorrect = 0;

        for (int j = 0; j < data.length; j++) {
            network.setInputValues(data[j].getData());
            network.run();

            Vector actual = data[j].getLabel().getData();
            Vector predicted = network.getOutputValues();


            boolean mismatch = !isEqualOutputs(actual, predicted);

            if (mismatch) {
                num_incorrect += 1;
            }
        }

        double error = num_incorrect / data.length * 100;
        return error;

    }


    /**
     * Compares two bit vectors to see if expected bit vector is most likely to be the same
     * class as the actual bit vector
     *
     * @param actual
     * @param predicted
     * @return
     */
    private static boolean isEqualOutputs(Vector actual, Vector predicted) {

        int max_at = 0;
        double max = 0;

//    Where the actual max should be
        int actual_index = 0;

        for (int i = 0; i < actual.size(); i++) {
            double aVal = actual.get(i);

            if (aVal == 1.0) {
                actual_index = i;
            }

            double bVal = predicted.get(i);

            if (bVal > max) {
                max = bVal;
                max_at = i;
            }
        }

        return actual_index == max_at;

    }


    /**
     * Reads a file formatted as CSV. Takes the labels and adds them to the set of labels (which
     * later helps determine the length of bit vectors). Records real-valued attributes. Turns the
     * attributes and labels into bit-vectors. Initializes a DataSet object with these instances.
     */
    private static void initializeInstances() {

        double[][] attributes = new double[num_examples][];

        String[] labels = new String[num_examples];
        unique_labels = new HashSet<>();


//    Reading dataset
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(FILENAME)));

//      You don't need these headers, they're just the column labels

            String useless_headers = br.readLine();

            for (int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[num_attributes];

                for (int j = 0; j < num_attributes; j++) {
                    attributes[i][j] = Double.parseDouble(scan.next());
                }

//        This last element is actually your classification, which is assumed to be a string
                labels[i] = scan.next();
                unique_labels.add(labels[i]);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }


//    Creating a mapping of bitvectors. So "some classification" => [0, 1, 0, 0]
        int distinct_labels = unique_labels.size();
        outputLayer = distinct_labels;

        bitvector_mappings = new HashMap<>();

        int index = 0;
        for (String label : unique_labels) {
            double[] bitvect = new double[distinct_labels];

//      At index, set to 1 for a given string
            bitvect[index] = 1.0;
//      Increment which index will have a bit flipped in next classification
            index++;

            bitvector_mappings.put(label, bitvect);
        }

//    Replaces the label for each instance with the corresponding bit vector for that label
//    This works even for binary classification
        allInstances = new Instance[num_examples];
        for (int i = 0; i < attributes.length; i++) {
            double[] X = attributes[i];

            String label = labels[i];
            double[] bitvect = bitvector_mappings.get(label);

            Instance instance = new Instance(X);
            instance.setLabel(new Instance(bitvect));

            allInstances[i] = instance;
        }
    }


    /**
     * Print out the actual vs expected bit-vector. Used for debugging purposes only
     *
     * @param actual   what the example's actual bit vector looks like
     * @param expected what a network output as a bit vector
     */
    public static void printVectors(Vector actual, Vector expected) {
        System.out.print("Actual: [");
        for (int i = 0; i < actual.size(); i++) {
            System.out.printf(" %f", actual.get(i));
        }
        System.out.print(" ] \t Expected: [");

        for (int i = 0; i < expected.size(); i++) {
            System.out.printf(" %f", expected.get(i));
        }
        System.out.println(" ]");
    }


    /**
     * Takes all instances, and randomly orders them. Then, the first PERCENT_TRAIN percentage of
     * instances form the trainSet DataSet, and the remaining (1 - PERCENT_TRAIN) percentage of
     * instances form the testSet DataSet.
     */
    public static void makeTestTrainSets() {

        List<Instance> instances = new ArrayList<>();
        for (Instance instance : allInstances) {
            instances.add(instance);
        }

        Random rand = new Random(SEED);
        if (shouldRandomize) {
            Collections.shuffle(instances, rand);
        }

        int cutoff = (int) (instances.size() * PERCENT_TRAIN);

        List<Instance> trainInstances = instances.subList(0, cutoff);
        List<Instance> testInstances = instances.subList(cutoff, instances.size());

        splittedTrainSet = new DataSet[5];

        for(int i = 1; i <= 5; i++) {
            int splitSize = (int) (trainInstances.size() * (i / 5.0));
            List<Instance> splitInstances = trainInstances.subList(0, splitSize);
            Instance[] arr_splt = new Instance[splitInstances.size()];
            splittedTrainSet[i-1] = new DataSet(splitInstances.toArray(arr_splt));
        }

        Instance[] arr_trn = new Instance[trainInstances.size()];
        trainSet = new DataSet(trainInstances.toArray(arr_trn));

//    System.out.println("Train Set ")

        Instance[] arr_tst = new Instance[testInstances.size()];
        testSet = new DataSet(testInstances.toArray(arr_tst));

    }


    /**
     * Given a DataSet of training data, separate the instances into K nearly-equal-sized
     * partitions called folds for K-folds cross validation
     *
     * @param training, the training DataSet
     * @return a list of folds, where each fold is an Instance[]
     */
    public static List<Instance[]> kfolds(DataSet training) {

        Instance[] trainInstances = training.getInstances();

        List<Instance> instances = new ArrayList<>();
        for (Instance instance : trainInstances) {
            instances.add(instance);
        }

        List<Instance[]> folds = new ArrayList<>();

//    Number of values per fold
        int per_fold = (int) Math.floor((double) (instances.size()) / K);

        int start = 0;
        int end = per_fold;

        while (start < instances.size()) {


            List<Instance> foldList = null;

            if (end > instances.size()) {
                end = instances.size();
            }
            foldList = instances.subList(start, end);

            Instance[] fold = new Instance[foldList.size()];
            fold = foldList.toArray(fold);

            folds.add(fold);

            start = end + 1;
            end = start + per_fold;

        }
        return folds;
    }


    /**
     * Given a list of Instance[], this helper combines each arrays contents into one, single
     * output array
     *
     * @param instanceList the list of Instance[]
     * @return the combined array consisting of the contents of each Instance[] in instanceList
     */
    public static Instance[] combineInstances(List<Instance[]> instanceList) {
        List<Instance> combined = new ArrayList<>();

        for (Instance[] fold : instanceList) {

            for (Instance instance : fold) {
                combined.add(instance);
            }
        }

        Instance[] combinedArr = new Instance[combined.size()];
        combinedArr = combined.toArray(combinedArr);
        return combinedArr;
    }


    /**
     * Given a list of folds and an index, it will provide an Instance[] with the combined
     * instances from every fold except for the fold at the given index
     *
     * @param folds     the K-folds, a list of Instance[] used as folds for cross-validation
     * @param foldIndex the index of the fold to exclude. That fold is used as the validation set
     * @return the training folds combined into once Instance[]
     */
    public static Instance[] getTrainFolds(List<Instance[]> folds, int foldIndex) {
        List<Instance[]> trainFolds = new ArrayList<>(folds);
        trainFolds.remove(foldIndex);

        Instance[] trnfolds = combineInstances(trainFolds);
        return trnfolds;
    }


    /**
     * Given a list of folds and an index, it will provide an Instance[] to serve as a validation
     * set.
     *
     * @param folds     the K-folds, a list of Instance[] used as folds for cross-validation
     * @param foldIndex the index of the fold to use as the validation set
     * @return the validation set
     */
    public static Instance[] getValidationFold(List<Instance[]> folds, int foldIndex) {
        return folds.get(foldIndex);
    }


}
/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.List;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class PA14GradientDescent {
    public static void helpMessage() {
        Log.info("Usage:");
        Log.info("\tjava PA13GradientDescent <data set> <gradient descent type> <batch size> <loss function> <epochs> <bias> <learning rate> <optimizer type> <mu/eps> <decayRate/beta1> <beta2> <layer_size_1 ... layer_size_n>");
        Log.info("\t\tdata set can be: 'and', 'or' or 'xor', 'iris' or 'mushroom'");
        Log.info("\t\tgradient descent type can be: 'stochastic', 'minibatch' or 'batch'");
        Log.info("\t\tbatch size should be > 0. Will be ignored for stochastic or batch gradient descent");
        Log.info("\t\tloss function can be: 'l1_norm', 'l2_norm', 'svm' or 'softmax'");
        Log.info("\t\tepochs is an integer > 0");
        Log.info("\t\tbias is a double");
        Log.info("\t\tlearning rate is a double usually small and > 0");
        Log.info("\t\toptimizer tpye is the gradient descent optimization we are using: either 'nesterov', 'rmsprop', or 'adam'");
        Log.info("\t\tmu/eps is either the momentum (double < 1 with typical values of 0.5, 0.9, 0.95 and 0.99) if optimizer type is nesterov or the eps value [double between 10^(-4) and 10^(-8)] if the optimizer type is rmsprop or adam");
        Log.info("\t\tdecayRate/beta1 is either the decay rate (double < 1) if optimizer type is rmsprop or beta1 (double with typical value of 0.9) if the optimizer type is adam, it is ignored if the optimizer type is nesterov");
        Log.info("\t\tbeta2 is the beta 2 value (double with typical value of 0.999) if the optimizer type is adam, and is ignored if the optimizer type is nesterov or rmsprop");
        Log.info("\t\tlayer_size_1..n is a list of integers which are the number of nodes in each hidden layer");
    }

    public static void main(String[] arguments) {
        if (arguments.length < 12) {
            helpMessage();
            System.exit(1);
        }

        String dataSetName = arguments[0];
        String descentType = arguments[1];
        int batchSize = Integer.parseInt(arguments[2]);
        String lossFunctionName = arguments[3];
        int epochs = Integer.parseInt(arguments[4]);
        double bias = Double.parseDouble(arguments[5]);
        double learningRate = Double.parseDouble(arguments[6]);
        String optimizerType = arguments[7];
        double mu = Double.parseDouble(arguments[8]);
        double eps = mu;
        double decayRate = Double.parseDouble(arguments[9]);
        double beta1 = decayRate;
        double beta2 = Double.parseDouble(arguments[10]);

        int[] layerSizes = new int[arguments.length - 11]; // the remaining arguments are the layer sizes
        for (int i = 11; i < arguments.length; i++) {
            layerSizes[i - 11] = Integer.parseInt(arguments[i]);
        }

        //the and, or and xor datasets will have 1 output (the number of output columns)
        //but the iris and mushroom datasets will have the number of output classes
        int outputLayerSize = 0;

        DataSet dataSet = null;
        if (dataSetName.equals("and")) {
            dataSet = new DataSet("and data", "./datasets/and.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("or")) {
            dataSet = new DataSet("or data", "./datasets/or.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("xor")) {
            dataSet = new DataSet("xor data", "./datasets/xor.txt");
            outputLayerSize = dataSet.getNumberOutputs();
        } else if (dataSetName.equals("iris")) {
            //TODO: PA1-4: Make sure you implement the getInputMeans,
            //getInputStandardDeviations and normalize methods in
            //DataSet to get this to work.
            dataSet = new DataSet("iris data", "./datasets/iris.txt");
            double[] means = dataSet.getInputMeans();
            double[] stdDevs = dataSet.getInputStandardDeviations();
            Log.info("data set means: " + Arrays.toString(means));
            Log.info("data set standard deviations: " + Arrays.toString(stdDevs));
            dataSet.normalize(means, stdDevs);

            outputLayerSize = dataSet.getNumberClasses();
        } else if (dataSetName.equals("mushroom")) {
            dataSet = new DataSet("mushroom data", "./datasets/agaricus-lepiota.txt");
            outputLayerSize = dataSet.getNumberClasses();
        } else {
            Log.fatal("unknown data set : " + dataSetName);
            System.exit(1);
        }

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("l1_norm")) {
            Log.info("Using an L1_NORM loss function.");
            lossFunction = LossFunction.L1_NORM;
        } else if (lossFunctionName.equals("l2_norm")) {
            Log.info("Using an L2_NORM loss function.");
            lossFunction = LossFunction.L2_NORM;
        } else if (lossFunctionName.equals("svm")) {
            Log.info("Using an SVM loss function.");
            lossFunction = LossFunction.SVM;
        } else if (lossFunctionName.equals("softmax")) {
            Log.info("Using an SOFTMAX loss function.");
            lossFunction = LossFunction.SOFTMAX;
        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        NeuralNetwork nn = new NeuralNetwork(dataSet.getNumberInputs(), layerSizes, outputLayerSize, lossFunction);
        try {
            nn.connectFully();
        } catch (NeuralNetworkException e) {
            Log.fatal("ERROR connecting the neural network -- this should not happen!.");
            e.printStackTrace();
            System.exit(1);
        }

        //start the gradient descent
        try {
            Log.info("Starting " + descentType + " gradient descent!");
            if (descentType.equals("minibatch")) {
                Log.info(descentType + "(" + batchSize + "), " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu/eps:" + mu + ", decayRate/beta1:" + decayRate + ", beta2:" + beta2);
            } else {
                Log.info(descentType + ", " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu:" + mu);
            }

            nn.initializeRandomly(bias);
            
            //Declare all possible variables for nesterov, rmsprop, and adam
            
            //Arrays to store the velocity and velocityPrev values for Nesterov Momentum (as per slide 36 of the fourth lesson)
            double[] velocity = new double[nn.getNumberWeights()];
            double[] velocityPrev = new double[velocity.length];
            
            //Array to store the cache values for RMSProp (as per slide 46 of the fourth lesson)
            double[] cache = new double[velocity.length];
            
            //Arrays to store the m, mt, v, and vt values for bias - corrected ADAM (as per last slide of the fourth lesson)
            double[] m = new double[velocity.length];
            double[] mt = new double[velocity.length];
            double[] v = new double[velocity.length];
            double[] vt = new double[velocity.length];
            
            double bestError = 10000;
            double error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
            double accuracy = nn.calculateAccuracy(dataSet.getInstances());

            if (error < bestError) bestError = error;
            System.out.println("  " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make hte accuracy a percentage*/);

            for (int i = 0; i < epochs; i++) {

                if (descentType.equals("stochastic")) {
                    //Shuffle the training data
                    dataSet.shuffle();
                    //Iterate through each instance
                    for(Instance inst : dataSet.getInstances()){
                        //Grab the current weights    
                        double[] newWeights = nn.getWeights();
                        //Calculate the gradient using only the current instance
                        double[] gradient = nn.getGradient(inst);
                        //Calculate new weights
                        for(int j = 0; j < gradient.length; j++){
                            //Nesterov weight update
                            if(optimizerType.equals("nesterov")){
                                velocityPrev[j] = velocity[j]; //Store original velocity
                                velocity[j] = mu*velocity[j] - learningRate*gradient[j]; //Calculate next velocity
                                newWeights[j] += (1 + mu)*velocity[j] - mu*velocityPrev[j]; //Perform weight update
                            } 
                            //RMSProp weight update
                            else if(optimizerType.equals("rmsprop")){
                                cache[j] = decayRate*cache[j] + (1 - decayRate)*gradient[j]*gradient[j]; //Calculate Cache value
                                newWeights[j] -= (learningRate/Math.sqrt(cache[j] + eps))*gradient[j]; //Perform weight update
                            } 
                            //ADAM weight update
                            else if(optimizerType.equals("adam")) {
                                m[j] = beta1*m[j] + (1 - beta1)*gradient[j]; //Calculate m
                                mt[j] = m[j]/(1 - Math.pow(beta1, i + 1)); //Unbias m
                                v[j] = beta2*v[j] + (1 - beta2)*gradient[j]*gradient[j]; //Calculate v
                                vt[j] = v[j]/(1 - Math.pow(beta2, i + 1)); //Unbias v
                                newWeights[j] -= learningRate*mt[j]/(Math.sqrt(vt[j] + eps)); //Perform weight update
                            } 
                            //Unknown optimizer
                            else {
                                Log.fatal("unknown optimizer : " + optimizerType);
                                System.exit(1);
                            }
                        }
                        //Set new weights
                        nn.setWeights(newWeights);
                    }

                } else if (descentType.equals("minibatch")) {
                    //Shuffle the training data
                    dataSet.shuffle();
                    //Iterate through each batch
                    for(int index = 0; index < dataSet.getInstances().size(); index += batchSize){
                        //Grab the current weights
                        double[] newWeights = nn.getWeights();
                        //Calculate the gradient using the current batch
                        double[] gradient = nn.getGradient(dataSet.getInstances(index, batchSize));
                        //Calculate new weights
                        for(int j = 0; j < gradient.length; j++){
                            //Nesterov weight update
                            if(optimizerType.equals("nesterov")){
                                velocityPrev[j] = velocity[j]; //Store original velocity
                                velocity[j] = mu*velocity[j] - learningRate*gradient[j]; //Calculate next velocity
                                newWeights[j] += (1 + mu)*velocity[j] - mu*velocityPrev[j]; //Perform weight update
                            } 
                            //RMSProp weight update
                            else if(optimizerType.equals("rmsprop")){
                                cache[j] = decayRate*cache[j] + (1 - decayRate)*gradient[j]*gradient[j]; //Calculate Cache value
                                newWeights[j] -= (learningRate/Math.sqrt(cache[j] + eps))*gradient[j]; //Perform weight update
                            } 
                            //ADAM weight update
                            else if(optimizerType.equals("adam")) {
                                m[j] = beta1*m[j] + (1 - beta1)*gradient[j]; //Calculate m
                                mt[j] = m[j]/(1 - Math.pow(beta1, i + 1)); //Unbias m
                                v[j] = beta2*v[j] + (1 - beta2)*gradient[j]*gradient[j]; //Calculate v
                                vt[j] = v[j]/(1 - Math.pow(beta2, i + 1)); //Unbias v
                                newWeights[j] -= learningRate*mt[j]/(Math.sqrt(vt[j] + eps)); //Perform weight update
                            } 
                            //Unknown optimizer
                            else {
                                Log.fatal("unknown optimizer : " + optimizerType);
                                System.exit(1);
                            }
                        }
                        //Set new weights
                        nn.setWeights(newWeights);
                    }

                } else if (descentType.equals("batch")) {
                    //Grad current weights
                    double[] newWeights = nn.getWeights();
                    //Calculate gradient using full data set
                    double[] gradient = nn.getGradient(dataSet.getInstances());
                    //Calculate new weights
                    for(int j = 0; j < gradient.length; j++){
                        //Nesterov weight update
                        if(optimizerType.equals("nesterov")){
                            velocityPrev[j] = velocity[j]; //Store original velocity
                            velocity[j] = mu*velocity[j] - learningRate*gradient[j]; //Calculate next velocity
                            newWeights[j] += (1 + mu)*velocity[j] - mu*velocityPrev[j]; //Perform weight update
                        } 
                        //RMSProp weight update
                        else if(optimizerType.equals("rmsprop")){
                            cache[j] = decayRate*cache[j] + (1 - decayRate)*gradient[j]*gradient[j]; //Calculate Cache value
                            newWeights[j] -= (learningRate/Math.sqrt(cache[j] + eps))*gradient[j]; //Perform weight update
                        } 
                        //ADAM weight update
                        else if(optimizerType.equals("adam")) {
                            m[j] = beta1*m[j] + (1 - beta1)*gradient[j]; //Calculate m
                            mt[j] = m[j]/(1 - Math.pow(beta1, i + 1)); //Unbias m
                            v[j] = beta2*v[j] + (1 - beta2)*gradient[j]*gradient[j]; //Calculate v
                            vt[j] = v[j]/(1 - Math.pow(beta2, i + 1)); //Unbias v
                            newWeights[j] -= learningRate*mt[j]/(Math.sqrt(vt[j] + eps)); //Perform weight update
                        } 
                        //Unknown optimizer
                        else {
                            Log.fatal("unknown optimizer : " + optimizerType);
                            System.exit(1);
                        }
                    }
                    //Set new weights
                    nn.setWeights(newWeights);

                } else {
                    Log.fatal("unknown descent type: " + descentType);
                    helpMessage();
                    System.exit(1);
                }

                //Log.info("weights: " + Arrays.toString(nn.getWeights()));

                //at the end of each epoch, calculate the error over the entire
                //set of instances and print it out so we can see if we're decreasing
                //the overall error
                error = nn.forwardPass(dataSet.getInstances()) / dataSet.getNumberInstances();
                accuracy = nn.calculateAccuracy(dataSet.getInstances());

                if (error < bestError) bestError = error;
                System.out.println(i + " " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make hte accuracy a percentage*/);
            }

        } catch (NeuralNetworkException e) {
            Log.fatal("gradient descent failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}


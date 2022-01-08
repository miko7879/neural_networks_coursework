/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.List;

import data.DataSet;
import data.Instance;

import network.LossFunction;
import network.NeuralNetwork;
import network.NeuralNetworkException;

import util.Log;
import util.Vector;


public class MKTests {

    public static void main(String[] arguments) throws NeuralNetworkException {
        if (arguments.length != 1) {
            System.err.println("Invalid arguments, you must specify a loss function, usage: ");
            System.err.println("\tjava PA13Tests <loss function>");
            System.err.println("\tloss function options are: 'l1_norm' or 'l2_norm'");
            System.exit(1);
        }

        String lossFunctionName = arguments[0];

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("l1_norm")) {
            Log.info("Using an L1_NORM loss function.");
            lossFunction = LossFunction.L1_NORM;

        } else if (lossFunctionName.equals("l2_norm")) {
            Log.info("Using an L2_NORM loss function.");
            lossFunction = LossFunction.L2_NORM;

        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        //creating this data set should work correctly if the previous test passed.
        DataSet xorData = new DataSet("xor data", "./datasets/xor.txt");

        //Define the test instances
        List<Instance> instances = xorData.getInstances(0, 4);
        
        //Define the network
        NeuralNetwork nn = PA11Tests.createSmallNeuralNetwork(xorData, lossFunction);
        
        //Define the problem weights
        double[] probWeights = new double[]{0.6295314787038659, 0.27838121948739913, 0.009047403578529956, -0.22080473506597365, -0.3720547790463711, -0.9781482090788205, 0.8823919936192828, 0.6007929500754432, 0.09964769349918612, 0.8344531641260844, 0.3351457163434062, 0.32434756957243494, 0.026880233220181626, -0.3255367109169427, -0.20577674595524242, -0.44570972578858803, -0.026429463851200596, 0.8466057018816455, -0.04000299698576426};
        nn.setWeights(probWeights);
        
        //Get the numeric and analytic gradients
        double[] numericGradient = nn.getNumericGradient(instances);
        double[] backpropGradient = nn.getGradient(instances);
        
        if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient)) {
            System.out.println("[INFO] Numeric Grad: " + java.util.Arrays.toString(numericGradient));
            System.out.println("[INFO] Analytic Grad: " + java.util.Arrays.toString(backpropGradient));
            System.out.println("[INFO] Weights: " + java.util.Arrays.toString(probWeights));
        }
        
        double[] numericGrad = new double[probWeights.length];
        
        for(Instance i : instances){
            System.out.println("New Instance");
            numericGradient = nn.getNumericGradient(i);
            for(int j = 0; j < numericGrad.length; j++){
                numericGrad[j] += numericGradient[j];
            }
            double[] backpropGrad = nn.getGradient(i);
            System.out.println("[INFO] Numeric Grad: " + java.util.Arrays.toString(numericGradient));
            System.out.println("[INFO] Analytic Grad: " + java.util.Arrays.toString(backpropGrad));
        }
        
        System.out.println("Numeric Grad");
        System.out.println("[INFO] Numeric Grad: " + java.util.Arrays.toString(numericGrad));
        
        if (BasicTests.gradientsCloseEnough(numericGrad, backpropGradient)) {
            System.out.println("Passed!");
        }
        
        
     }

}


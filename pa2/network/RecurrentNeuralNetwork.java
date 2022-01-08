package network;

import java.util.List;

import data.Sequence;
import data.CharacterSequence;
import data.TimeSeries;

import util.Log;


public class RecurrentNeuralNetwork {
    //this is the loss function for the output of the neural network
    LossFunction lossFunction;
    
    //this is the maximum length of any sequence
    int maxSequenceLength;

    //this is the total number of weights in the neural network
    int numberWeights;
    
    //layers contains all the nodes in the neural network
    RecurrentNode[][] layers;

    public RecurrentNeuralNetwork(int inputLayerSize, int[] hiddenLayerSizes, int outputLayerSize, int maxSequenceLength, RNNNodeType rnnNodeType, LossFunction lossFunction) {
        this.lossFunction = lossFunction;
        this.maxSequenceLength = maxSequenceLength;

        //the number of layers in the neural network is 2 plus the number of hidden layers,
        //one additional for the input, and one additional for the output.

        //create the outer array of the 2-dimensional array of nodes
        layers = new RecurrentNode[hiddenLayerSizes.length + 2][];
        
        //we will progressively calculate the number of weights as we create the network. the
        //number of edges will be equal to the number of hidden nodes (each has a bias weight, but
        //the input and output nodes do not) plus the number of edges
        numberWeights = 0;

        Log.info("creating a neural network with " + hiddenLayerSizes.length + " hidden layers, for max sequence length: " + maxSequenceLength + ".");
        for (int layer = 0; layer < layers.length; layer++) {
            
            //determine the layer size depending on the layer number, 0 is the
            //input layer, and the last layer is the output layer, all others
            //are hidden layers
            int layerSize;
            NodeType nodeType;
            if (layer == 0) {
                //this is the input layer
                layerSize = inputLayerSize;
                nodeType = NodeType.INPUT;
                Log.info("input layer " + layer + " has " + layerSize + " nodes.");

            } else if (layer < layers.length - 1) {
                //this is a hidden layer
                layerSize = hiddenLayerSizes[layer - 1];
                nodeType = NodeType.HIDDEN;
                Log.info("hidden layer " + layer + " has " + layerSize + " nodes.");

            } else {
                //this is the output layer
                layerSize = outputLayerSize;
                nodeType = NodeType.OUTPUT;
                Log.info("output layer " + layer + " has " + layerSize + " nodes.");
            }

            //create the layer with the right length and right node types
            layers[layer] = new RecurrentNode[layerSize];

            for (int j = 0; j < layers[layer].length; j++) {
                if (nodeType == NodeType.INPUT) {
                    //input nodes do not have an activation function applied to them
                    layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.LINEAR);

                } else {
                    switch (rnnNodeType) {
                        case LINEAR:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.LINEAR);
                            //increment the number of weights here because some node types will have a different amount
                            //linear nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case SIGMOID:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.SIGMOID);
                            //increment the number of weights here because some node types will have a different amount
                            //sigmoid nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case TANH:
                            layers[layer][j] = new RecurrentNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength, ActivationType.TANH);
                            //increment the number of weights here because some node types will have a different amount
                            //tanh nodes don't have bias in the output layer
                            if (nodeType == NodeType.HIDDEN) numberWeights++;
                            break;
                        case LSTM:
                            //TODO: implement this for Programming Assignment 2 - Part 4
                            //increment the number of weights here because some node types will have a different amount
                            layers[layer][j] = new LSTMNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 11;
                            break;
                        case GRU:
                            layers[layer][j] = new GRUNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 9;
                            break;
                        case MGU:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            break;
                        case UGRNN:
                            //TODO: BONUS implement this for Programming Assignment 2 - Part 4
                            break;
                        case DELTA:
                            layers[layer][j] = new DeltaNode(layer, j /*i is the node number*/, nodeType, maxSequenceLength);
                            numberWeights += 6;
                            break;
                        default:
                            Log.fatal("Trying to create RNN with unknown RNNNodeType - this should never happen!");
                            System.exit(1);
                    }
                }
            }
        }
    }

    /**
     * This gets the number of weights in the RecurrentNeuralNetwork, which should
     * be equal to the number of hidden nodes (1 bias per hidden node) plus 
     * the number of edges (1 bias per edge). It is updated whenever an edge 
     * is added to the neural network.
     *
     * @return the number of weights in the neural network.
     */
    public int getNumberWeights() {
        return numberWeights;
    }

    /**
     * This resets all the values that are modified in the forward pass and 
     * backward pass and need to be reset to 0 before doing another
     * forward and backward pass (i.e., all the non-weights/biases).
     */
    public void reset() {
        for (int layer = 0; layer < layers.length; layer++) {
            for (int number = 0; number < layers[layer].length; number++) {
                //call reset on each node in the network
                layers[layer][number].reset();
            }
        }
    }

    /**
     * Gets the name for each weight so things can be debugged easier
     * @return a array of strings, each corresponding to what edge/bias each weight represents from the
     *      results of the getWeights() method.
     */
    public String[] getWeightNames() throws NeuralNetworkException {
        String[] weightNames = new String[numberWeights];

        //What we're going to do here is fill in the weights array
        //we just created by having each node set the weights starting
        //at the position variable we're creating. The Node.getWeights
        //method will set the weights variable passed as a parameter,
        //and then return the number of weights it set. We can then
        //use this to increment position so the next node gets weights
        //and puts them in the right position in the weights array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].getWeightNames(position, weightNames);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the weight names there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weightNames;
    }


    /**
     * This returns an array of every weight (including biases) in the RecurrentNeuralNetwork.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public double[] getWeights() throws NeuralNetworkException {
        double[] weights = new double[numberWeights];

        //What we're going to do here is fill in the weights array
        //we just created by having each node set the weights starting
        //at the position variable we're creating. The Node.getWeights
        //method will set the weights variable passed as a parameter,
        //and then return the number of weights it set. We can then
        //use this to increment position so the next node gets weights
        //and puts them in the right position in the weights array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].getWeights(position, weights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the RecurrentNeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking. 
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the RecurrentNeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the RecurrentNeuralNetwork.
     * This will be very useful in backpropagation and sanity checking.
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public double[] getDeltas() throws NeuralNetworkException {
        double[] deltas = new double[numberWeights];

        //What we're going to do here is fill in the deltas array
        //we just created by having each node set the deltas starting
        //at the position variable we're creating. The Node.getDeltas
        //method will set the deltas variable passed as a parameter,
        //and then return the number of deltas it set. We can then
        //use this to increment position so the next node gets deltas
        //and puts them in the right position in the deltas array.
        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nDeltas = layers[layer][nodeNumber].getDeltas(position, deltas);
                position += nDeltas;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the RecurrentNeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This adds edges to the RecurrentNeuralNetwork, connecting each node
     * in a layer to each node in the subsequent layer
     */
    public void connectFully() throws NeuralNetworkException {
        //create outgoing edges from the input layer to the last hidden layer,
        //the output layer will not have outgoing edges
        for (int layer = 0; layer < layers.length - 1; layer++) {

            //iterate over the nodes in the current layer
            for (int inputNodeNumber = 0; inputNodeNumber < layers[layer].length; inputNodeNumber++) {

                //iterate over the nodes in the next layer
                for (int outputNodeNumber = 0; outputNodeNumber < layers[layer + 1].length; outputNodeNumber++) {
                    RecurrentNode inputNode = layers[layer][inputNodeNumber];
                    RecurrentNode outputNode = layers[layer + 1][outputNodeNumber];
                    new Edge(inputNode, outputNode);

                    //as we added an edge, the number of weights should increase by 1
                    numberWeights++;
                    Log.trace("numberWeights now: " + numberWeights);
                }
            }
        }
    }

    /**
     * Makes this RNN a Jordan network by creating a RecurrentEdge between
     * every output node and every hidden node.
     *
     * @param timeSkip is how many time steps to skip for the recurrent connection
     */
    public void connectJordan(int timeSkip) throws NeuralNetworkException {
        for(int t = 1; t <= timeSkip; t++){
            for(int outNode = 0; outNode < layers[layers.length - 1].length; outNode++){
                for(int layer = 1; layer < layers.length - 1; layer++){
                    for(int hidNode = 0; hidNode < layers[layer].length; hidNode++){
                        RecurrentNode hiddenNode = layers[layer][hidNode];
                        RecurrentNode outputNode = layers[layers.length - 1][outNode];
                        new RecurrentEdge(outputNode, hiddenNode, t);
                        numberWeights++;
                        Log.trace("numberWeights now: " + numberWeights);
                    }
                }
            }
        }

    }

    /**
     * Makes this RNN an Elman network by creating a RecurrentEdge from 
     * every hidden node in a layer to every other hidden node in that layer
     *
     * @param timeSkip is how many time steps to skip for the recurrent connection
     */
    public void connectElman(int timeSkip) throws NeuralNetworkException {
        for(int t = 1; t <= timeSkip; t++){
            for(int hidLayer = 1; hidLayer < layers.length - 1; hidLayer++){
                RecurrentNode[] hiddenLayer = layers[hidLayer];
                for(int hidNode = 0; hidNode < hiddenLayer.length; hidNode++){
                    RecurrentNode hiddenNode = hiddenLayer[hidNode];
                    for(int partNode = 0; partNode < hiddenLayer.length; partNode++){
                        RecurrentNode partnerNode = hiddenLayer[partNode];
                        new RecurrentEdge(hiddenNode, partnerNode, t);
                        numberWeights++;
                        Log.trace("numberWeights now: " + numberWeights);
                    }
                }
            }
        }

    }


    /**
     * This will create an Edge between the node with number inputNumber on the inputLayer to the
     * node with the outputNumber on the outputLayer.
     *
     * @param inputLayer the layer of the input node
     * @param inputNumber the number of the input node on layer inputLayer
     * @param outputLayer the layer of the output node
     * @param outputNumber the number of the output node on layer outputLayer
     */
    public void connectNodes(int inputLayer, int inputNumber, int outputLayer, int outputNumber) throws NeuralNetworkException {
        if (inputLayer >= outputLayer) {
            throw new NeuralNetworkException("Cannot create an Edge between input layer " + inputLayer + " and output layer " + outputLayer + " because the layer of the input node must be less than the layer of the output node.");
        //} else if (outputLayer != inputLayer + 1) {
            //throw new NeuralNetworkException("Cannot create an Edge between input layer " + inputLayer + " and output layer " + outputLayer + " because the layer of the output node must be the next layer in the network.");
        }
        RecurrentNode inputNode = layers[inputLayer][inputNumber];
        RecurrentNode outputNode = layers[outputLayer][outputNumber];
        new Edge(inputNode, outputNode);
        numberWeights++;
        Log.trace("numberWeights now: " + numberWeights);
    }

    /**
     * This initializes the weights in the RNN using either Xavier or
     * Kaiming initialization.
    *
     * @param type will be either "xavier" or "kaiming" and this will
     * initialize the child nodes accordingly, using their helper methods.
     * @param bias is the value to set the bias of each node to.
     */
    public void initializeRandomly(String type, double bias) throws NeuralNetworkException {
        if(type.equals("xavier")){
            for(int layer = 1; layer < layers.length; layer++){
                int fanIn = 0;
                int fanOut = 0;
                //Calculate fan in and fan out of layer
                for(int node = 0; node < layers[layer].length; node++){
                    fanIn += layers[layer][node].fanIn();
                    fanOut += layers[layer][node].fanOut();
                }
                //Initialize weights
                for(int node = 0; node < layers[layer].length; node++){
                    layers[layer][node].initializeWeightsAndBiasXavier(fanIn, fanOut, bias);
                }
            }
        } else if(type.equals("kaiming")){
            for(int layer = 1; layer < layers.length; layer++){
                int fanIn = 0;
                int fanOut = 0;
                //Calculate fan in and fan out of layer
                for(int node = 0; node < layers[layer].length; node++){
                    fanIn += layers[layer][node].fanIn();
                }
                //Initialize weights
                for(int node = 0; node < layers[layer].length; node++){
                    layers[layer][node].initializeWeightsAndBiasKaiming(fanIn, bias);
                }
            }
        } else {
            throw new NeuralNetworkException("Unknown initialization method: " + type);
        }
        

    }



    /**
     * This performs a forward pass through the neural network given
     * inputs from the input instance.
     *
     * @param instance is the data set instance to pass through the network
     *
     * @return the sum of the output of all output nodes
     */
    public double forwardPass(Sequence sequence) throws NeuralNetworkException {
        //be sure to reset before doing a forward pass
        reset();

        //set input values differently for time series and character sequences
        if (sequence instanceof CharacterSequence) {
            CharacterSequence characterSequence = (CharacterSequence)sequence;

            //set the input nodes for each time step in the CharacterSequence
            for (int timeStep = 0; timeStep < characterSequence.getLength() - 1; timeStep++) {
                int value = characterSequence.valueAt(timeStep);
                for (int number = 0; number < layers[0].length; number++) {
                    RecurrentNode inputNode = layers[0][number];
                    if (value == number) {
                        //the value will be 0..n where n is the number of input nodes (and possible character values)
                        inputNode.preActivationValue[timeStep] = 1.0;
                    } else {
                        inputNode.preActivationValue[timeStep] = 0.0;
                    }
                }
            }

        } else if (sequence instanceof TimeSeries) {
            TimeSeries series = (TimeSeries)sequence;

            //set the input nodes for each time step in the TimeSeries
            for (int timeStep = 0; timeStep < series.getLength() - 1; timeStep++) {
                for (int number = 0; number < layers[0].length; number++) {
                    RecurrentNode inputNode = layers[0][number];
                    inputNode.preActivationValue[timeStep] = series.getInputValue(timeStep, number);
                }
            }

        }
        
        //Propogate each 
        for(int t = 0; t < sequence.getLength(); t++){
            for(int layer = 0; layer < layers.length; layer++){
                for(int node = 0; node < layers[layer].length; node++){
                    layers[layer][node].propagateForward(t);
                }
            }
        }





        //The following is needed for Programming Assignment 2 - Part 1
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        //note that the target value for any time step is the sequence value at that time step + 1
        //this means you should only go up to length - 1 time steps in calculating the loss
        double lossSum = 0;
        if (sequence instanceof CharacterSequence) {
            //calculate the loss functions for character sequences
            CharacterSequence characterSequence = (CharacterSequence)sequence;

            if (lossFunction == LossFunction.NONE) {
                //for no loss function we can just return the sum of the outputs overall time steps, and can set each output node's delta to 1
                for (int timeStep = 0; timeStep < characterSequence.getLength() - 1; timeStep++) {
                    for (int number = 0; number < nOutputs; number++) {
                        RecurrentNode outputNode = layers[outputLayer][number];
                        lossSum += outputNode.postActivationValue[timeStep];
                        outputNode.delta[timeStep] = 1.0;
                    }
                }

            } else if (lossFunction == LossFunction.SVM) {
                for(int t = 0; t < characterSequence.getLength() - 1; t++) {
                    double wrongOutputDeltaSum = 0;
                    int correctNode = characterSequence.valueAt(t + 1);
                    for(int outNode = 0; outNode < nOutputs; outNode++){
                        if(outNode != correctNode){
                            RecurrentNode outputNode = layers[outputLayer][outNode];
                            double testVal = outputNode.postActivationValue[t] - layers[outputLayer][correctNode].postActivationValue[t] + 1;
                            if(testVal > 0)
                                lossSum += testVal;
                            outputNode.delta[t] += (testVal < 0) ? 0 : 1.0;
                            wrongOutputDeltaSum -= outputNode.delta[t];
                        }
                    }
                    //Calculate the incoming delta of the correct output node
                    layers[outputLayer][correctNode].delta[t] += wrongOutputDeltaSum;        
                }

            } else if (lossFunction == LossFunction.SOFTMAX) {
                for(int t = 0; t < characterSequence.getLength() - 1; t++) {
                    double softmaxSum = 0;
                    double maxZ = layers[outputLayer][0].postActivationValue[t];
                    int correctNode = characterSequence.valueAt(t + 1);
                    //Find the maximum output
                    for(int outNode = 1; outNode < nOutputs; outNode++){
                        RecurrentNode outputNode = layers[outputLayer][outNode];
                        if(outputNode.postActivationValue[t] > maxZ)
                            maxZ = outputNode.postActivationValue[t];
                    }
                    //Calculate the SOFTMAX sum
                    for(int outNode = 0; outNode < nOutputs; outNode++){
                        RecurrentNode outputNode = layers[outputLayer][outNode];
                        softmaxSum += Math.exp(outputNode.postActivationValue[t] - maxZ);
                    }
                    //Calculate the deltas for each node and the output sum
                    for(int outNode = 0; outNode < nOutputs; outNode++){
                        RecurrentNode outputNode = layers[outputLayer][outNode];
                        //Baseline delta
                        outputNode.delta[t] += Math.exp(outputNode.postActivationValue[t] - maxZ)/softmaxSum;
                        //When y = j set the output sum and adjust the delta
                        if(outNode == correctNode){
                            lossSum -= Math.log(outputNode.delta[t]);
                            outputNode.delta[t] -= 1;
                        }
                    }
                }

            } else {
                throw new NeuralNetworkException("Could not do a CharacterSequence forward pass on RecurrentNeuralNetwork because lossFunction was unknown or invalid: " + lossFunction);
            }
        } else if (sequence instanceof TimeSeries) {
            TimeSeries series = (TimeSeries)sequence;

            if (lossFunction == LossFunction.NONE) {
                //for no loss function we can just return the sum of the outputs overall time steps, and can set each output node's delta to 1
                for (int timeStep = 0; timeStep < series.getLength() - 1; timeStep++) {
                    for (int number = 0; number < nOutputs; number++) {
                        RecurrentNode outputNode = layers[outputLayer][number];
                        lossSum += outputNode.postActivationValue[timeStep];
                        outputNode.delta[timeStep] = 1.0;
                    }
                }

            } else if (lossFunction == LossFunction.L1_NORM) {
                for (int t = 0; t < series.getLength() - 1; t++) {
                    for (int outNode = 0; outNode < nOutputs; outNode++) {
                        RecurrentNode outputNode = layers[outputLayer][outNode];
                        lossSum += Math.abs(outputNode.postActivationValue[t] - series.getOutputValue(t + 1, outNode));
                        outputNode.delta[t] += (outputNode.postActivationValue[t] < series.getOutputValue(t + 1, outNode)) ? -1.0 : 1.0;
                    }
                }

            } else if (lossFunction == LossFunction.L2_NORM) {
                for (int t = 0; t < series.getLength() - 1; t++) {
                    double outputSum = 0;
                    for(int outNode = 0; outNode < nOutputs; outNode++) {
                        RecurrentNode outputNode = layers[outputLayer][outNode];
                        outputSum += (outputNode.postActivationValue[t] - series.getOutputValue(t + 1, outNode))*(outputNode.postActivationValue[t] - series.getOutputValue(t + 1, outNode));
                    }
                    outputSum = Math.sqrt(outputSum);
                    for (int outNode = 0; outNode < nOutputs; outNode++) {
                        RecurrentNode outputNode = layers[outputLayer][outNode];
                        outputNode.delta[t] += (outputNode.postActivationValue[t] - series.getOutputValue(t + 1, outNode))/outputSum;
                    }
                    lossSum += outputSum;
                }

            } else {
                throw new NeuralNetworkException("Could not do a TimeSeries forward pass on RecurrentNeuralNetwork because lossFunction was unknown: " + lossFunction);
            }
        }

        return lossSum;
    }

    /**
     * This performs multiple forward passes through the neural network
     * by multiple instances are returns the output sum.
     *
     * @param sequences is the set of CharacterSequences to pass through the network
     *
     * @return the sum of their outputs
     */
    public double forwardPass(List<Sequence> sequences) throws NeuralNetworkException {
        double sum = 0.0;

        for (Sequence sequence : sequences) {
            sum += forwardPass(sequence);
        }

        return sum;
    }

    /**
     * This performs multiple forward passes through the recurrent neural network
     * and calculates how many of the character predictions for every
     * sequence were classified correctly.
     *
     * @param sequences is the set of CharacterSequences to pass through the network
     *
     * @return a percentage (between 0 and 1) of how many character predictions were
     * correctly classified
     */
    public double calculateAccuracy(List<CharacterSequence> sequences) throws NeuralNetworkException {
        //Counters to keep track of total attempts and correct guesses
        int numCorrect = 0;
        int numTotal = 0;
        //Grab the output layer
        RecurrentNode[] outputLayer = layers[layers.length - 1];
        //Iterate through each sequence
        for(CharacterSequence s : sequences){
            //Calculate the number of instances in this sequence, last one doesn't count
            int numInst = s.getLength() - 1;
            //Add it to the number of total guesses
            numTotal += numInst;
            //Pass the sequence through the neural network
            forwardPass(s);
            for(int t = 0; t < numInst; t ++){
                //The correct value is the value at the next timestep
                int correctNode = s.valueAt(t + 1);
                //Find the predicted node
                int predictedNode = 0;
                double maxValue = outputLayer[0].postActivationValue[t];
                for(int i = 1; i < outputLayer.length; i++){
                    double testVal = outputLayer[i].postActivationValue[t];
                    if(testVal > maxValue){
                        predictedNode = i;
                        maxValue = testVal;
                    }
                }
                //Compare the predicted value with the correct one and if it matches then increment the correct counter
                numCorrect += ((predictedNode == correctNode) ? 1 : 0);
            }
        }
        //Return the accuracy
        return (double)numCorrect/(double)numTotal;
    }


    /**
     * This gets the output values of the neural network 
     * after a forward pass, this will be a 2 dimensional array for
     * an RNN because we'll have outputs for each time step
     *
     * @param seuence is the CharacterSequence which generated the the output values, we need this to get the length of the sequence
     *
     * @return a two dimensional array of the output values from this neural network for each time step
     */
    public double[][] getOutputValues(Sequence sequence) {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[][] outputValues = new double[sequence.getLength() - 1][nOutputs];

        //we will have 1 less output because we're predicting one character ahead
        for (int timeStep = 0; timeStep < sequence.getLength() - 1; timeStep++) {
            for (int number = 0; number < nOutputs; number++) {
                outputValues[timeStep][number] = layers[outputLayer][number].postActivationValue[timeStep];
            }
        }
        return outputValues;
    }

    /**
     * The step size used to calculate the gradient numerically using the finite
     * difference method.
     */
    private static final double H = 0.0000001;

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     */
    public double[] getNumericGradient(Sequence sequence) throws NeuralNetworkException {
        //Grab the weights
        double[] weights = getWeights();
        //Create an array to store the numeric gradient
        double[] numericGrad = new double[weights.length];
        //Iterate through each weight
        for(int weight = 0; weight < weights.length; weight++){
            //Calculate upper value of finite difference
            weights[weight] += H;
            setWeights(weights);
            double fPlus = forwardPass(sequence);
            //Calculate lower value of finite difference
            weights[weight] = weights[weight] - H - H;
            setWeights(weights);
            double fMinus = forwardPass(sequence);
            //Do the finite difference calculation
            numericGrad[weight] = (fPlus - fMinus)/2.0/H;
            //Restore weight values
            weights[weight] += H;
        }
        //Restore Original Weights
        setWeights(weights);
        //Return the numeric gradient
        return numericGrad;
    }

    /**
     * This calculates the gradient of the neural network with it's current
     * weights for a given DataSet Instance using the finite difference method:
     * gradient[i] = (f(x where x[i] = x[i] + H) - f(x where x[i] = x[i] - H)) / 2h
     */
    public double[] getNumericGradient(List<Sequence> sequences) throws NeuralNetworkException {
        //Create an array to store the numeric gradient
        double[] numericGrad = new double[getWeights().length];
        for(Sequence s : sequences){
            double[] instanceGrad = getNumericGradient(s);
            for(int j = 0; j < numericGrad.length; j++){
                numericGrad[j] += instanceGrad[j];
            }
        }
        return numericGrad;    
    }


    /**
     * This performs a backward pass through the neural network given 
     * outputs from the given instance. This will set the deltas in
     * all the edges and nodes which will be used to calculate the 
     * gradient and perform backpropagation.
     *
     */
    public void backwardPass(Sequence sequence) throws NeuralNetworkException {
        //TODO: You need to implement this for Programming Assignment 2 - Part 2
        //note you should start at sequence.getLength() - 2 (not -1) as this is the last
        //time step that was passed forward through the RNN
        for(int t = sequence.getLength() - 2; t >= 0; t--){
            for(int layer = layers.length - 1; layer >=0; layer--){
                for(int node = layers[layer].length - 1; node >= 0; node--){
                    layers[layer][node].propagateBackward(t);
                }
            }
        }
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the RecurrentNeuralNetwork.backwardPass(Sequence)) Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param sequence is the training CharacterSequence for the forward and 
     * backward pass.
     */
    public double[] getGradient(Sequence sequence) throws NeuralNetworkException {
        forwardPass(sequence);
        backwardPass(sequence);

        return getDeltas();
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the RecurrentNeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient). The resulting gradient should be the sum of
     * each delta for each instance.
     *
     * @param sequences are the training CharacterSequences for the forward and 
     * backward passes.
     */
    public double[] getGradient(List<Sequence> sequences) throws NeuralNetworkException {
        //Initialize an array to store the gradient	
        double[] gradient = new double[getWeights().length];
        //Go through each instance
        for(Sequence s : sequences){
            //Calculate the gradient using a forward/backward pass
            forwardPass(s);
            backwardPass(s);
            double[] iDelta = getDeltas();
            //Add it to the running tally of the gradients
            for(int j = 0; j < gradient.length; j++){
                gradient[j] += iDelta[j];
            }
        }
        //Return the running tally
        return gradient;
    }
}

package network;

import java.util.List;

import data.Instance;

import util.Log;


public class NeuralNetwork {
    //this is the loss function for the output of the neural 
    //network, you will use this in PA1-3
    LossFunction lossFunction;
    

    //this is the total number of weights in the neural network
    int numberWeights;
    
    //layers contains all the nodes in the neural network
    Node[][] layers;

    public NeuralNetwork(int inputLayerSize, int[] hiddenLayerSizes, int outputLayerSize, LossFunction lossFunction) {
        this.lossFunction = lossFunction;

        //the number of layers in the neural network is 2 plus the number of hidden layers,
        //one additional for the input, and one additional for the output.

        //create the outer array of the 2-dimensional array of nodes
        layers = new Node[hiddenLayerSizes.length + 2][];
        
        //we will progressively calculate the number of weights as we create the network. the
        //number of edges will be equal to the number of hidden nodes (each has a bias weight, but
        //the input and output nodes do not) plus the number of edges
        numberWeights = 0;

        Log.info("creating a neural network with " + hiddenLayerSizes.length + " hidden layers.");
        for (int layer = 0; layer < layers.length; layer++) {
            
            //determine the layer size depending on the layer number, 0 is the
            //input layer, and the last layer is the output layer, all others
            //are hidden layers
            int layerSize;
            NodeType nodeType;
            ActivationType activationType;
            if (layer == 0) {
                //this is the input layer
                layerSize = inputLayerSize;
                nodeType = NodeType.INPUT;
                activationType = ActivationType.LINEAR;
                Log.info("input layer " + layer + " has " + layerSize + " nodes.");

            } else if (layer < layers.length - 1) {
                //this is a hidden layer
                layerSize = hiddenLayerSizes[layer - 1];
                nodeType = NodeType.HIDDEN;
                activationType = ActivationType.TANH;
                Log.info("hidden layer " + layer + " has " + layerSize + " nodes.");

                //increment the number of weights by the number of nodes in
                //this hidden layer
                numberWeights += layerSize; 
            } else {
                //this is the output layer
                layerSize = outputLayerSize;
                nodeType = NodeType.OUTPUT;
                activationType = ActivationType.SIGMOID;
                Log.info("output layer " + layer + " has " + layerSize + " nodes.");
            }

            //create the layer with the right length and right node types
            layers[layer] = new Node[layerSize];
            for (int j = 0; j < layers[layer].length; j++) {
                layers[layer][j] = new Node(layer, j /*i is the node number*/, nodeType, activationType);
            }
        }
    }

    /**
     * This gets the number of weights in the NeuralNetwork, which should
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
     * This returns an array of every weight (including biases) in the NeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when getting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return weights;
    }

    /**
     * This sets every weight (including biases) in the NeuralNetwork, it sets them in
     * the same order that they are retreived by the getWeights method.
     * This will be very useful in backpropagation and sanity checking. 
     *
     * @throws NeuralNetworkException if numberWeights was not calculated correctly. This shouldn't happen.
     */
    public void setWeights(double[] newWeights) throws NeuralNetworkException {
        if (numberWeights != newWeights.length) {
            throw new NeuralNetworkException("Could not setWeights because the number of new weights: " + newWeights.length + " was not equal to the number of weights in the NeuralNetwork: " + numberWeights);
        }

        int position = 0;
        for (int layer = 0; layer < layers.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++) {
                int nWeights = layers[layer][nodeNumber].setWeights(position, newWeights);
                position += nWeights;

                if (position > numberWeights) {
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when setting the weights there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }
    }

    /**
     * This returns an array of every weight (including biases) in the NeuralNetwork.
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
                    throw new NeuralNetworkException("The numberWeights field of the NeuralNetwork was (" + numberWeights + ") but when getting the deltas there were more hidden nodes and edges than numberWeights. This should not happen unless numberWeights is not being updated correctly.");
                }
            }
        }

        return deltas;
    }


    /**
     * This adds edges to the NeuralNetwork, connecting each node
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
                    Node inputNode = layers[layer][inputNodeNumber];
                    Node outputNode = layers[layer + 1][outputNodeNumber];
                    new Edge(inputNode, outputNode);

                    //as we added an edge, the number of weights should increase by 1
                    numberWeights++;
                    Log.trace("numberWeights now: " + numberWeights);
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
        if (inputLayer >= outputLayer)
            throw new NeuralNetworkException("Cannot create an Edge between input layer " + inputLayer + " and output layer " + outputLayer + " because the layer of the input node must be less than the layer of the output node.");
	//Grab the input node
	Node inputNode = layers[inputLayer][inputNumber];
	//Grab the output node
	Node outputNode = layers[outputLayer][outputNumber];
	//Connect the two by creating a new edge, this will add the edge to the input and output nodes' respective edge lists
	new Edge(inputNode, outputNode);
	//Increment the number of weights
	numberWeights++;
	//Log the change
	Log.trace("numberWeights now: " + numberWeights);
    }

    /**
     * This initializes the weights properly by setting the incoming
     * weights for each edge using a random normal distribution (i.e.,
     * a gaussian distribution) and dividing the randomly generated
     * weight by sqrt(n) where n is the fan-in of the node. It also
     * sets the bias for each node to the given parameter.
     *
     * For example, if we have a node N which has 5 input edges,
     * the weights of each of those edges will be generated by
     * Random.nextGaussian()/sqrt(5). The best way to do this is
     * to iterate over each node and have it use the 
     * Node.initializeWeightsAndBias(double bias) method.
     *
     * @param bias is the value to set the bias of each node to.
     */
    public void initializeRandomly(double bias) {
        //Go through all layers starting with the first hidden layer (no incoming edges or bias for input nodes)
	for(int layer = 1; layer < layers.length; layer++){
	    for(int node = 0; node < layers[layer].length; node++){
	        layers[layer][node].initializeWeightsAndBias(bias);
	    }
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
    public double forwardPass(Instance instance) throws NeuralNetworkException {
        //Reset before doing a forward pass
        reset();
	
	//Go through each node layer - by - layer order
	for(int layer = 0; layer < layers.length; layer++){
	    for(int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++){
	        if(layer == 0) //Initialize the input nodes
		    layers[layer][nodeNumber].preActivationValue = instance.inputs[nodeNumber];
		layers[layer][nodeNumber].propagateForward(); //Activate the node and propogate the value forward
	    }
	}
	
        //The following is needed for PA1-3 and PA1-4
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;
        double outputSum = 0;

	//No loss function
        if (lossFunction == LossFunction.NONE) {
            //Just sum up the outputs
            for (int number = 0; number < nOutputs; number++) {
                Node outputNode = layers[outputLayer][number];
                outputSum += outputNode.postActivationValue;
                outputNode.delta = 1.0;
            }
        }

	//L1 Norm Loss Function
	else if (lossFunction == LossFunction.L1_NORM) {
            for(int number = 0; number < nOutputs; number++){
		Node outputNode = layers[outputLayer][number];
		outputSum += Math.abs(outputNode.postActivationValue - instance.expectedOutputs[number]);
		outputNode.delta = (outputNode.postActivationValue < instance.expectedOutputs[number]) ? -1.0 : 1.0;
	    }
        }

	//L2 Norm Loss Function
	else if (lossFunction == LossFunction.L2_NORM) {
            for(int number = 0; number < nOutputs; number++){
	    	Node outputNode = layers[outputLayer][number];
		outputSum += (outputNode.postActivationValue - instance.expectedOutputs[number])*(outputNode.postActivationValue - instance.expectedOutputs[number]);
	    }
	    outputSum = Math.sqrt(outputSum);
	    for(int number = 0; number < nOutputs; number++){
	    	Node outputNode = layers[outputLayer][number];
		outputNode.delta = (outputNode.postActivationValue - instance.expectedOutputs[number])/outputSum;
	    }

        } 

	//SVM Loss Function
	else if (lossFunction == LossFunction.SVM) {
	    //Declare variable to store the sum of deltas for cases where j =/= y
	    double wrongOutputDeltaSum = 0;
	    //Grab the expected output
	    int correctNode = (int)instance.expectedOutputs[0];
	    //Calculate the output sum and incoming deltas for each output node where j =/= y
            for(int number = 0; number < nOutputs; number++){
		if(number != correctNode){
		    Node outputNode = layers[outputLayer][number];
		    double testVal = outputNode.postActivationValue - layers[outputLayer][correctNode].postActivationValue + 1;
		    if(testVal > 0)
		    	outputSum += testVal;
		    outputNode.delta = (testVal < 0) ? 0 : 1.0;
		    wrongOutputDeltaSum -= outputNode.delta;
		}
	    }
	    //Calculate the incoming delta of the correct output node
	    layers[outputLayer][correctNode].delta = wrongOutputDeltaSum;
        } 
	
	//SOFTMAX Loss Function
	else if (lossFunction == LossFunction.SOFTMAX) {
            //Declare variable to store the SOFTMAX sum
	    double softmaxSum = 0;
	    //Declare variable to store the max output, initially the first node is assumed to be the largest
	    double maxZ = layers[outputLayer][0].postActivationValue;
	    //Grab the expected output
	    int correctNode = (int)instance.expectedOutputs[0];
	    //Find the maximum output
	    for(int number = 1; number < nOutputs; number++){
		Node outputNode = layers[outputLayer][number];
		if(outputNode.postActivationValue > maxZ)
		    maxZ = outputNode.postActivationValue;
	    }
	    //Calculate the SOFTMAX sum
	    for(int number = 0; number < nOutputs; number++){
		Node outputNode = layers[outputLayer][number];
		softmaxSum += Math.exp(outputNode.postActivationValue - maxZ);
	    }
	    //Calculate the deltas for each node and the output sum
	    for(int number = 0; number < nOutputs; number++){
		Node outputNode = layers[outputLayer][number];
		//Baseline delta
		outputNode.delta = Math.exp(outputNode.postActivationValue - maxZ)/softmaxSum;
		//When y = j set the output sum and adjust the delta
		if(number == correctNode){
		    outputSum -= Math.log(outputNode.delta);
		    outputNode.delta -= 1;
		}
	    }
        } 

	//Unknown Loss Function
	else {
            throw new NeuralNetworkException("Could not do forward pass on NeuralNetwork because lossFunction was unknown: " + lossFunction);
        }

	//Return the output sum
        return outputSum;
    }

    /**
     * This performs multiple forward passes through the neural network
     * by multiple instances are returns the output sum.
     *
     * @param instances is the set of instances to pass through the network
     *
     * @return the sum of their outputs
     */
    public double forwardPass(List<Instance> instances) throws NeuralNetworkException {
        double sum = 0.0;

        for (Instance instance : instances) {
            sum += forwardPass(instance);
        }

        return sum;
    }

    /**
     * This performs multiple forward passes through the neural network
     * and calculates how many of the instances were classified correctly.
     *
     * @param instances is the set of instances to pass through the network
     *
     * @return a percentage (between 0 and 1) of how many instances were
     * correctly classified
     */
    public double calculateAccuracy(List<Instance> instances) throws NeuralNetworkException {
        //Variable to store the number of correct guesses
	int numCorrect = 0;
	//Determine the output layer
	int outputLayer = layers.length - 1;
	//Go through instance by isntance
	for(Instance i : instances){
	    //Grab the expected value
	    int correctNode = (int)i.expectedOutputs[0];
	    //Pass the instance through the network and grab the outputs
	    forwardPass(i);
	    double[] outputs = getOutputValues();
	    //Initially the first node is assumed correct
	    int predictedNode = 0;
	    double maxVal = outputs[0];
	    //Determine the output node
	    for(int j = 1; j < outputs.length; j++){
		if(outputs[j] > maxVal){
		    maxVal = outputs[j];
		    predictedNode = j;
		}
	    }
	    //If the output node matched the expected node, increment number of correct guesses
	    if(predictedNode == correctNode)
		numCorrect++;
	}
	//Calculate the proportion of correct guesses and return
	return (double) numCorrect / (double) instances.size();
    }


    /**
     * This gets the output values of the neural network 
     * after a forward pass.
     *
     * @return an array of the output values from this neural network
     */
    public double[] getOutputValues() {
        //the number of output values is the number of output nodes
        int outputLayer = layers.length - 1;
        int nOutputs = layers[outputLayer].length;

        double[] outputValues = new double[nOutputs];

        for (int number = 0; number < nOutputs; number++) {
            outputValues[number] = layers[outputLayer][number].postActivationValue;
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
    public double[] getNumericGradient(Instance instance) throws NeuralNetworkException {
	//Grab the weights
	double[] weights = getWeights();
	//Create an array to store the numeric gradient
	double[] numericGrad = new double[weights.length];
	//Iterate through each weight
	for(int weight = 0; weight < weights.length; weight++){
	    //Calculate upper value of finite difference
	    weights[weight] += H;
	    setWeights(weights);
	    double fPlus = forwardPass(instance);
	    //Calculate lower value of finite difference
	    weights[weight] = weights[weight] - H - H;
	    setWeights(weights);
	    double fMinus = forwardPass(instance);
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
    public double[] getNumericGradient(List<Instance> instances) throws NeuralNetworkException {
	//Create an array to store the numeric gradient
	double[] numericGrad = new double[getWeights().length];
    	for(Instance i : instances){
	    double[] instanceGrad = getNumericGradient(i);
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
    public void backwardPass(Instance instance) throws NeuralNetworkException {
	//Traverse the network layer - by - layer
	for(int layer = layers.length - 1; layer >= 0; layer--){
	    for(int nodeNumber = 0; nodeNumber < layers[layer].length; nodeNumber++){
		layers[layer][nodeNumber].propagateBackward(); //Propagate the gradient backwards
	    }
	}
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the NeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient).
     *
     * @param instance is the training instance/sample for the forward and 
     * backward pass.
     */
    public double[] getGradient(Instance instance) throws NeuralNetworkException {
        forwardPass(instance);
        backwardPass(instance);
        return getDeltas();
    }

    /**
     * This gets the gradient of the neural network at its current
     * weights and the given instance using backpropagation (e.g., 
     * the NeuralNetwork.backwardPass(Instance))* Method.
     *
     * Helpful tip: use getDeltas after doing the propagateBackwards through
     * the networks to get the gradients/deltas in the same order as the
     * weights (which will be the same order as they're calculated for
     * the numeric gradient). The resulting gradient should be the sum of
     * each delta for each instance.
     *
     * @param instances are the training instances/samples for the forward and 
     * backward passes.
     */
    public double[] getGradient(List<Instance> instances) throws NeuralNetworkException {
	//Initialize an array to store the gradient	
	double[] gradient = new double[getWeights().length];
	//Go through each instance
	for(Instance i : instances){
	    //Calculate the gradient using a forward/backward pass
	    forwardPass(i);
	    backwardPass(i);
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

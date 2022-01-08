/**
 * This class represents a Node in the neural network. It will
 * have a list of all input and output edges, as well as its
 * own value. It will also track it's layer in the network and
 * if it is an input, hidden or output node.
 */
package network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import util.Log;

public class Node {
    //the layer of this node in the neural network. This is set 
    //final so we cannot change it after assigning it (which 
    //could cause bugs)
    final int layer;

    //the number of this node in it's layer. this is mostly
    //just used for printing friendly error message
    final int number;

    //the type of this node (input, hidden or output). This 
    //is set final so we cannot change it after assigning it 
    //(which could cause bugs)
    final NodeType nodeType;

    //the activation function this node will use, can be
    //either sigmoid, tanh or softmax (for the output
    //layer).
    final ActivationType activationType;

    //this is the value which is calculated by the forward
    //pass (if it is hidden or output), or assigned by the
    //data set if it is an input node, before the activation
    //function is applied
    public double preActivationValue;

    //this is the value which is calculated by the forward
    //pass (if it is not an input node) after the activation
    //function has been applied.
    public double postActivationValue;

    //this is the delta/error calculated by backpropagation
    public double delta;

    //this is the bias value added to the sum of the inputs
    //multiplied by the weights before the activation function
    //is applied
    private double bias;

    //thius is the delta/error calculated by backpropagation
    //for the bias
    private double biasDelta;

    //this is a list of all incoming edges to this node
    private List<Edge> inputEdges;

    //this is a list of all outgoing edges from this node
    private List<Edge> outputEdges;


    /**
     * This creates a new node at a given layer in the
     * network and specifies it's type (either input,
     * hidden, our output).
     *
     * @param layer is the layer of the Node in
     * the neural network
     * @param type is the type of node, specified by
     * the Node.NodeType enumeration.
     */
    public Node(int layer, int number, NodeType nodeType, ActivationType activationType) {
        this.layer = layer;
        this.number = number;
        this.nodeType = nodeType;
        this.activationType = activationType;

        //assign its value to 0.
        preActivationValue = 0;
        postActivationValue = 0;

        //initialize the input and output edges lists
        //as ArrayLists
        inputEdges = new ArrayList<Edge>();
        outputEdges = new ArrayList<Edge>();

        Log.trace("Created a node: " + toString());
    }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        Log.trace("Resetting node: " + toString());

        preActivationValue = 0;
        postActivationValue = 0;
        delta = 0;
        biasDelta = 0;

        for (Edge outputEdge : outputEdges) {
            outputEdge.weightDelta = 0;
        }
    }


    /**
     * The Edge class will call this on its output Node when it is
     * constructed (the input and output nodes of an edge
     * are passed as parameters to the Edge constructor).
     *
     * @param outgoingEdge the new outgoingEdge to add
     *
     * @throws NeuralNetworkException if the edge already exists
     */
    public void addOutgoingEdge(Edge outgoingEdge) throws NeuralNetworkException {
        //lets have a sanity check to make sure we don't duplicate adding
        //an edge
        for (Edge edge : outputEdges) {
            if (edge.equals(outgoingEdge)) {
                throw new NeuralNetworkException("Attempted to add an outgoing edge to node " + toString()
                        + " but could not as it already had an edge to the same output node: " + edge.outputNode.toString());
            }
        }

        Log.trace("Node " + toString() + " added outgoing edge to Node " + outgoingEdge.outputNode);
        outputEdges.add(outgoingEdge);
    }

    /**
     * The Edge class will call this on its input Node when it is
     * constructed (the input and output nodes of an edge
     * are passed as parameters to the Edge constructor).
     *
     * @param incomingEdge the new incomingEdge to add
     *
     * @throws NeuralNetworkException if the edge already exists
     */
    public void addIncomingEdge(Edge incomingEdge) throws NeuralNetworkException {
        //lets have a sanity check to make sure we don't duplicate adding
        //an edge
        for (Edge edge : inputEdges) {
            if (edge.equals(incomingEdge)) {
                throw new NeuralNetworkException("Attempted to add an incoming edge to node " + toString()
                        + " but could not as it already had an edge to the same input node: " + edge.inputNode.toString());
            }
        }

        Log.trace("Node " + toString() + " added incoming edge from Node " + incomingEdge.inputNode);
        inputEdges.add(incomingEdge);
    }


    /**
     * Used to get the weights of this node along with the weights
     * of all of it's outgoing edges. It will set the weights in the weights
     * parameter passed in starting at position, and return the number of
     * weights it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            weights[position] = bias;
            weightCount = 1;
        }

        for (Edge edge : outputEdges) {
            weights[position + weightCount] = edge.weight;
            weightCount++;
        }

        return weightCount;
    }

    /**
     * Used to get the deltas of this node along with the deltas
     * of all of it's outgoing edges. It will set the deltas in the deltas
     * parameter passed in starting at position, and return the number of
     * deltas it set.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        //the first delta set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            deltas[position] = biasDelta;
            deltaCount = 1;
        }

        for (Edge edge : outputEdges) {
            deltas[position + deltaCount] = edge.weightDelta;
            deltaCount++;
        }

        return deltaCount;
    }


    /**
     * Used to set the weights of this node along with the weights of
     * all it's outgoing edges. It uses the same technique as Node.getWeights
     * where the starting position of weights to set is passed, and it returns
     * how many weights were set.
     * 
     * @param position is the starting position in the weights parameter to start
     * setting weights from.
     * @param weights is the array of weights we are setting from
     *
     * @return the number of weights gotten from the weights parameter
     */

    public int setWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType == NodeType.HIDDEN) {
            bias = weights[position];
            weightCount = 1;
        }

        for (Edge edge : outputEdges) {
            edge.weight = weights[position + weightCount];
            weightCount++;
        }

        return weightCount;
    }

    /**
     * This will evaluate the sigmoid function at the given value.
     */
    private double sigmoid(double val){
	return 1.0/(1.0 + Math.exp(-1.0*val));
    }

    /**
     * This will evaluate the tanh function at the given value.
     */
    private double tanh(double val){
	return Math.tanh(val);
    }

    /**
     * This applys the linear activation function to this node. The postActivationValue
     * will be set to the preActivationValue.
     */
    public void applyLinear() {
	postActivationValue = preActivationValue;
    }

    /**
     * This applys the sigmoid function to this node. The postActivationValue
     * will be set to sigmoid(preActivationValue).
     */
    public void applySigmoid() {
	postActivationValue = sigmoid(preActivationValue);
    }

    /**
     * This applys the tanh function to this node. The postActivationValue
     * will be set to tanh(preActivationValue).
     */
    public void applyTanh() {
	postActivationValue = tanh(preActivationValue);
    }

    /**
     * This applys the softmax function to this node. The postActivationValue
     * will be set to softmax(preActivationValue). This requires passing
     * the sum of the output layer values (i.e., the bottom of the softmax function
     * fraction.)
     *
     * @param softmaxSum the sum of all e^(outputs) - the bottom of the softmax function
     * fraction
     */
    public void applySoftmax(double softmaxSum) {
        //TODO: YOu need to implement this for Programming Assignment 1 - Part 3
    }

    /**
     * This applies the specified activation function to this node. The function to use is
     * specified on node instantiation by the activationType parameter.
     */
    public void activate() throws NeuralNetworkException {
	if(activationType == ActivationType.LINEAR)
	   applyLinear();
	else if(activationType == ActivationType.SIGMOID)
	   applySigmoid();
	else if(activationType == ActivationType.TANH)
	   applyTanh();
	else
	   throw new NeuralNetworkException("Unknown activation type: " + activationType);
    }
  
    /**
     * This propagates the postActivationValue at this node
     * to all it's output nodes.
     */
    public void propagateForward() throws NeuralNetworkException {
	//Add the bias
	if(nodeType == NodeType.HIDDEN)
	   preActivationValue += bias;
	//Activate this node
	activate();
	//Propogate the value to all downstream nodes
	for(Edge e : outputEdges){
	   e.outputNode.preActivationValue += postActivationValue*e.weight;
	}
    }

    /**
     * This calculated the derivative based on the activation function. The function to use is
     * specified on node instantiation by the activationType parameter.
     */
    public double derivative(double val) throws NeuralNetworkException {
	//The derivative of a linear activation function is 1
	if(activationType == ActivationType.LINEAR)
	   return 1.0;
	//The derivative of sigmoid(x) is sigmoid(x)*[1 - sigmoid(x)]
	else if(activationType == ActivationType.SIGMOID)
	   return sigmoid(val)*(1.0 - sigmoid(val));
	//The derivative of tanh(x) is 1 - [tanh(x)]^2
	else if(activationType == ActivationType.TANH)
	   return 1.0 - tanh(val)*tanh(val);
	//Otherwise the derivative is unknown, will expand this if/else block when more activation functions are introduced
	else
	   throw new NeuralNetworkException("Unknown activation type: " + activationType);
    }

    /**
     * This propagates the delta back from this node
     * to its incoming edges.
     */
    public void propagateBackward() throws NeuralNetworkException {
	//The gradient of this node is the derivative of the activation function evaluated at the input value times the incoming gradient
	delta = delta*derivative(preActivationValue);
	//The gradient of the bias is the same as the gradient of the node
	biasDelta = delta;
	//Propagate the node's gradient to the incoming edges
	for(Edge e : inputEdges){
	    e.propagateBackward(delta);
	}
    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly initializes each incoming edge weight by using
     *  Random.nextGaussian() / sqrt(N) where N is the number
     *  of incoming edges.
     *
     *  @param bias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBias(double bias) {
	Random weightGen = new Random();
	double fanInFactor = Math.sqrt(inputEdges.size());
        //If hidden node then set the bias
	if(nodeType == NodeType.HIDDEN)
	    this.bias = bias;
	//Randomly initialize all incoming edges
	for(Edge e : inputEdges){
	    e.weight = weightGen.nextGaussian()/fanInFactor;
	}
    }

    /**
     * Prints concise information about this node.
     *
     * @return The node as a short string.
     */
    public String toString() {
        return "[Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }

    /**
     * Prints detailed information about this node.
     *
     * @return The node as a detailed string.
     */
    public String toDetailedString() {
        return "[Node - layer: " + layer + ", number: " + number + ", node type: " + nodeType + ", activation type: " + activationType + ", n input edges: " + inputEdges.size() + ", n output edges: " + outputEdges.size()  + ", pre value: " + preActivationValue + ", post value: " + postActivationValue + ", delta: " + delta + "]";
    }
    
}

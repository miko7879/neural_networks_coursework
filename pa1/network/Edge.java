/**
 * This class represents an Edge in a neural network. It will contain
 * the Edge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import util.Log;

public class Edge {
    //the weight for this edge
    public double weight;

    //the delta calculated by backpropagation for this edge
    public double weightDelta;

    //the input node of this edge
    public Node inputNode;

    //the output node of this edge
    public Node outputNode;

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingEdge(Edge) and Node.addIncomingEdge(Edge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public Edge(Node inputNode, Node outputNode) throws NeuralNetworkException {
        this.inputNode = inputNode;
        this.outputNode = outputNode;
        Log.debug("Created a new edge with input " + inputNode.toString() + " and output " + outputNode.toString());

        //initialize the weight and delta to 0
        weight = 0.0;
        weightDelta = 0.0;

        inputNode.addOutgoingEdge(this);
        outputNode.addIncomingEdge(this);
    }

    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public void propagateBackward(double delta) {
	//The gradient of this weight is equal to the incoming gradient times the forward pass output of the incoming edge
	weightDelta = delta*inputNode.postActivationValue;
	//The contribution of this edge to the incoming gradient of the source node is the incoming gradient to the edge times the weight
	inputNode.delta += delta*weight;
    }

}

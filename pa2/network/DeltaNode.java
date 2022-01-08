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

public class DeltaNode extends RecurrentNode {

    //these are the weight values for the Delta node
    double v;
    double a;
    double b1;
    double b2;

    //these are the bias values for the Delta node
    double rb;
    double zb;

    //these are the deltas for the weights and biases
    public double delta_v;
    public double delta_a;
    public double delta_b1;
    public double delta_b2;
    public double delta_rb;
    public double delta_zb;

    //this is the delta value for rt in the diagram
    public double[] rt;

    //variable C saved for doing the backward pass
    public double[] C;


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
    public DeltaNode(int layer, int number, NodeType nodeType, int maxSequenceLength) {
        super(layer, number, nodeType, maxSequenceLength, null);
        rt = new double[maxSequenceLength];
        C = new double[maxSequenceLength];
    }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        //use RecurrentNode's reset to reset everything this has inherited from
        //RecurrentNode, then reset the DeltaNode's fields
        super.reset();
        Log.trace("Resetting Delta node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            rt[timeStep] = 0;
            C[timeStep] = 0;
        }

        delta_v = 0;
        delta_a = 0;
        delta_b1 = 0;
        delta_b2 = 0;
        delta_rb = 0;
        delta_zb = 0;
    }


    /**
     * We need to override the getWeightNames from RecurrentNode as
     * a DeltaNode will have 6 weight and bias names as opposed to
     * just one bias.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weightNames is the array of weight nameswe're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeightNames(int position, String[] weightNames) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            
            weightNames[position] = "Delta Node [layer " + layer + ", number " + number + ", v]";
            weightNames[position + 1] = "Delta Node [Layer " + layer + ", number " + number + ", a]";
            weightNames[position + 2] = "Delta Node [Layer " + layer + ", number " + number + ", b1]";
            weightNames[position + 3] = "Delta Node [Layer " + layer + ", number " + number + ", b2]";
            weightNames[position + 4] = "Delta Node [Layer " + layer + ", number " + number + ", rb]";
            weightNames[position + 5] = "Delta Node [Layer " + layer + ", number " + number + ", zb]";

            weightCount += 6;
        }

        for (Edge edge : outputEdges) {
            String targetType = "";
            if (edge.outputNode instanceof LSTMNode) targetType = "LSTM ";
            if (edge.outputNode instanceof GRUNode) targetType = "GRU ";
            if (edge.outputNode instanceof DeltaNode) targetType = "Delta ";
            weightNames[position + weightCount] = "Edge from Delta Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            String targetType = "";
            if (recurrentEdge.outputNode instanceof LSTMNode) targetType = "LSTM ";
            if (recurrentEdge.outputNode instanceof GRUNode) targetType = "GRU ";
            if (recurrentEdge.outputNode instanceof DeltaNode) targetType = "Delta ";
            weightNames[position + weightCount] = "Recurrent Edge from Delta Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }


        return weightCount;
    }



    /**
     * We need to override the getWeights from RecurrentNode as
     * a DeltaNode will have 6 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public int getWeights(int position, double[] weights) {
        int weightCount = 0;

        //the first weight set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            
            weights[position] = v;
            weights[position + 1] = a;
            weights[position + 2] = b1;
            weights[position + 3] = b2;
            weights[position + 4] = rb;
            weights[position + 5] = zb;

            weightCount += 6;
        }

        for (Edge edge : outputEdges) {
            weights[position + weightCount] = edge.weight;
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            weights[position + weightCount] = recurrentEdge.weight;
            weightCount++;
        }

        return weightCount;
    }

    /**
     * We need to override the getDeltas from RecurrentNode as
     * an DeltaNode will have 6 weights and biases as opposed to
     * just one bias.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public int getDeltas(int position, double[] deltas) {
        int deltaCount = 0;

        //the first delta set will be the bias if it is a hidden node
        if (nodeType != NodeType.INPUT) {
            
            deltas[position] = delta_v;
            deltas[position + 1] = delta_a;
            deltas[position + 2] = delta_b1;
            deltas[position + 3] = delta_b2;
            deltas[position + 4] = delta_rb;
            deltas[position + 5] = delta_zb;

            deltaCount += 6;
        }

        for (Edge edge : outputEdges) {
            deltas[position + deltaCount] = edge.weightDelta;
            deltaCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            deltas[position + deltaCount] = recurrentEdge.weightDelta;
            deltaCount++;
        }

        return deltaCount;
    }


    /**
     * We need to override the getDeltas from RecurrentNode as
     * a DeltaNode will have 6 weights and biases as opposed to
     * just one bias.
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
        if (nodeType != NodeType.INPUT) {
            
            v = weights[position];
            a = weights[position + 1];
            b1 = weights[position + 2];
            b2 = weights[position + 3];
            rb = weights[position + 4];
            zb = weights[position + 5];

            weightCount += 6;
        }

        for (Edge edge : outputEdges) {
            edge.weight = weights[position + weightCount];
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            recurrentEdge.weight = weights[position + weightCount];
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
     * This propagates the postActivationValue at this GRU node
     * to all it's output nodes.
     */
    public void propagateForward(int timeStep) {
        
        //Get the previous output
        double z_prev = (timeStep == 0) ? 0 : postActivationValue[timeStep - 1];
        
        //Calculate rt
        rt[timeStep] = sigmoid(preActivationValue[timeStep] + rb);
        
        //Calculate C, which simplifies the backward pass
        C[timeStep] = tanh(zb + preActivationValue[timeStep]*b2 + preActivationValue[timeStep]*a*v*z_prev + v*z_prev*b1);
        
        //Calculate the output
        postActivationValue[timeStep] = tanh(C[timeStep]*(1 - rt[timeStep]) + rt[timeStep]*z_prev);
        
        //Go through each regular edge and propagate it forwards
        for(Edge e : outputEdges){
            e.outputNode.preActivationValue[timeStep] += postActivationValue[timeStep]*e.weight;
        }
        
        //Go through each recurrent edge and propagate it forwards
        for(RecurrentEdge re : outputRecurrentEdges){
            if(timeStep + re.timeSkip < preActivationValue.length)
                re.outputNode.preActivationValue[timeStep + re.timeSkip] += postActivationValue[timeStep]*re.weight;
        }
        
    }

    /**
     * This propagates the delta back from this node
     * to its incoming edges.
     */
    public void propagateBackward(int timeStep) {
        
        //The previous output is used in the backward pass
        double z_prev = (timeStep == 0) ? 0 : postActivationValue[timeStep - 1];
        
        //The derivative of rt
        double d_rt = rt[timeStep]*(1.0 - rt[timeStep]);
        
        //The derivative of zt
        double d_zt = 1.0 - postActivationValue[timeStep]*postActivationValue[timeStep];
        
        //The derivative of C
        double d_C = 1.0 - C[timeStep]*C[timeStep];
        
        //Update the weights
        delta_v += z_prev*(a*preActivationValue[timeStep] + b1)*d_C*(1 - rt[timeStep])*d_zt*delta[timeStep];
        delta_a += z_prev*v*preActivationValue[timeStep]*d_C*(1 - rt[timeStep])*d_zt*delta[timeStep];
        delta_b1 += z_prev*v*d_C*(1 - rt[timeStep])*d_zt*delta[timeStep];
        delta_b2 += preActivationValue[timeStep]*d_C*(1 - rt[timeStep])*d_zt*delta[timeStep];
        
        //Update the biases
        delta_rb += d_rt*(z_prev - C[timeStep])*d_zt*delta[timeStep];
        delta_zb += d_C*(1 - rt[timeStep])*d_zt*delta[timeStep];
        
        //Add to the delta of the previous timestep
        if(timeStep != 0){
            delta[timeStep - 1] += v*(a*preActivationValue[timeStep] + b1)*d_C*(1 - rt[timeStep])*d_zt*delta[timeStep]
                                + rt[timeStep]*d_zt*delta[timeStep];
        }
        
        //Set the outgoing delta for this node
        delta[timeStep] = b2*d_C*(1 - rt[timeStep])*d_zt*delta[timeStep]
                        + a*v*z_prev*d_C*(1 - rt[timeStep])*d_zt*delta[timeStep]
                        + d_rt*(z_prev - C[timeStep])*d_zt*delta[timeStep];
        
        //Propogate the node's gradient to incoming edges
        for(Edge e : inputEdges)
            e.propagateBackward(timeStep, delta[timeStep]);
        for(RecurrentEdge r : inputRecurrentEdges)
            r.propagateBackward(timeStep, delta[timeStep]);

    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly initializes each incoming edge weight by using
     *  Random.nextGaussian() / sqrt(N) where N is the number
     *  of incoming edges.
     *
     *  @param bias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasKaiming(int fanIn, double bias) {
        
        Random weightGen = new Random();
        
        v = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        a = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        b1 = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        b2 = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        rb = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        zb = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        
        for(Edge e : inputEdges)
            e.weight = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        
        for(RecurrentEdge re: inputRecurrentEdges)
            re.weight = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        
    }

    /**
     *  This sets the node's bias to the bias parameter and then
     *  randomly intializes each incoming edge weight uniformly
     *  at random (you can use Random.nextDouble()) between 
     *  +/- sqrt(6) / sqrt(fan_in + fan_out) 
     *
     *  @param bias is the bias to initialize this node's bias to
     */
    public void initializeWeightsAndBiasXavier(int fanIn, int fanOut, double bias) {
        
        Random weightGen = new Random();
        
        v = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        a = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        b1 = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        b2 = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        rb = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        zb = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        
        for(Edge e : inputEdges)
            e.weight = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        
        for(RecurrentEdge re: inputRecurrentEdges)
            re.weight = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        
    }


    /**
     * Prints concise information about this node.
     *
     * @return The node as a short string.
     */
    public String toString() {
        return "[Delta Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}

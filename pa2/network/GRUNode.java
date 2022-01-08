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

public class GRUNode extends RecurrentNode {

    //these are the weight values for the GRU node
    double hw;
    double rw;
    double zw;

    double hu;
    double ru;
    double zu;

    //these are the bias values for the GRU node
    double hb;
    double rb;
    double zb;

    //these are the deltas for the weights and biases
    public double delta_hw;
    public double delta_rw;
    public double delta_zw;

    public double delta_hu;
    public double delta_ru;
    public double delta_zu;

    public double delta_hb;
    public double delta_rb;
    public double delta_zb;

    //this is the delta value for rt in the diagram
    public double[] rt;

    //this is the delta value for zt in the diagram
    public double[] zt;

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
    public GRUNode(int layer, int number, NodeType nodeType, int maxSequenceLength) {
        super(layer, number, nodeType, maxSequenceLength, null);
        rt = new double[maxSequenceLength];
        zt = new double[maxSequenceLength];
        C = new double[maxSequenceLength];
    }

    /**
     * This resets the values which need to be recalcualted for
     * each forward and backward pass. It will also reset the
     * deltas for outgoing nodes.
     */
    public void reset() {
        //use RecurrentNode's reset to reset everything this has inherited from
        //RecurrentNode, then reset the GRUNode's fields
        super.reset();
        Log.trace("Resetting GRU node: " + toString());

        for (int timeStep = 0; timeStep < maxSequenceLength; timeStep++) {
            rt[timeStep] = 0;
            zt[timeStep] = 0;
            C[timeStep] = 0;
        }

        delta_hw = 0;
        delta_rw = 0;
        delta_zw = 0;

        delta_hu = 0;
        delta_ru = 0;
        delta_zu = 0;

        delta_hb = 0;
        delta_rb = 0;
        delta_zb = 0;
    }


    /**
     * We need to override the getWeightNames from RecurrentNode as
     * a GRUNode will have 9 weight and bias names as opposed to
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
            
            weightNames[position] = "GRU Node [layer " + layer + ", number " + number + ", hw]";
            weightNames[position + 1] = "GRU Node [Layer " + layer + ", number " + number + ", rw]";
            weightNames[position + 2] = "GRU Node [Layer " + layer + ", number " + number + ", zw]";
            
            weightNames[position + 3] = "GRU Node [Layer " + layer + ", number " + number + ", hu]";
            weightNames[position + 4] = "GRU Node [Layer " + layer + ", number " + number + ", ru]";
            weightNames[position + 5] = "GRU Node [Layer " + layer + ", number " + number + ", zu]";
            
            weightNames[position + 6] = "GRU Node [Layer " + layer + ", number " + number + ", hb]";
            weightNames[position + 7] = "GRU Node [Layer " + layer + ", number " + number + ", rb]";
            weightNames[position + 8] = "GRU Node [Layer " + layer + ", number " + number + ", zb]";

            weightCount += 9;
        }

        for (Edge edge : outputEdges) {
            String targetType = "";
            if (edge.outputNode instanceof LSTMNode) targetType = "LSTM ";
            if (edge.outputNode instanceof GRUNode) targetType = "GRU ";
            if (edge.outputNode instanceof DeltaNode) targetType = "Delta ";
            weightNames[position + weightCount] = "Edge from GRU Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + edge.outputNode.layer + ", number " + edge.outputNode.number + "]";
            weightCount++;
        }

        for (RecurrentEdge recurrentEdge : outputRecurrentEdges) {
            String targetType = "";
            if (recurrentEdge.outputNode instanceof LSTMNode) targetType = "LSTM ";
            if (recurrentEdge.outputNode instanceof GRUNode) targetType = "GRU ";
            if (recurrentEdge.outputNode instanceof DeltaNode) targetType = "Delta ";
            weightNames[position + weightCount] = "Recurrent Edge from GRU Node [layer " + layer + ", number " + number + "] to " + targetType + "Node [layer " + recurrentEdge.outputNode.layer + ", number " + recurrentEdge.outputNode.number + "]";
            weightCount++;
        }


        return weightCount;
    }



    /**
     * We need to override the getWeights from RecurrentNode as
     * a GRUNode will have 9 weights and biases as opposed to
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
            
            weights[position] = hw;
            weights[position + 1] = rw;
            weights[position + 2] = zw;
            
            weights[position + 3] = hu;
            weights[position + 4] = ru;
            weights[position + 5] = zu;
            
            weights[position + 6] = hb;
            weights[position + 7] = rb;
            weights[position + 8] = zb;

            weightCount += 9;
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
     * an GRUNode will have 9 weights and biases as opposed to
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
            
            deltas[position] = delta_hw;
            deltas[position + 1] = delta_rw;
            deltas[position + 2] = delta_zw;
            
            deltas[position + 3] = delta_hu;
            deltas[position + 4] = delta_ru;
            deltas[position + 5] = delta_zu;
            
            deltas[position + 6] = delta_hb;
            deltas[position + 7] = delta_rb;
            deltas[position + 8] = delta_zb;

            deltaCount += 9;
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
     * a GRUNode will have 9 weights and biases as opposed to
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
            
            hw = weights[position];
            rw = weights[position + 1];
            zw = weights[position + 2];
            
            hu = weights[position + 3];
            ru = weights[position + 4];
            zu = weights[position + 5];
            
            hb = weights[position + 6];
            rb = weights[position + 7];
            zb = weights[position + 8];

            weightCount += 9;
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
        double h_prev = (timeStep == 0) ? 0 : postActivationValue[timeStep - 1];
        
        //Calculate rt
        rt[timeStep] = sigmoid(rw*preActivationValue[timeStep] + ru*h_prev + rb);
        
        //Calculate zt
        zt[timeStep] = sigmoid(zw*preActivationValue[timeStep] + zu*h_prev + zb);
        
        //Calculate C, which simplifies the backward pass
        C[timeStep] = tanh(hu*h_prev*rt[timeStep] + hb + hw*preActivationValue[timeStep]);
        
        //Calculate the output
        postActivationValue[timeStep] = C[timeStep]*(1 - zt[timeStep]) + zt[timeStep]*h_prev;
        
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
        double h_prev = (timeStep == 0) ? 0 : postActivationValue[timeStep - 1];
        
        //The derivative of rt
        double d_rt = rt[timeStep]*(1.0 - rt[timeStep]);
        
        //The derivative of zt
        double d_zt = zt[timeStep]*(1.0 - zt[timeStep]);
        
        //The derivative of C
        double d_C = 1.0 - C[timeStep]*C[timeStep];
        
        //Update the input weights
        delta_hw += preActivationValue[timeStep]*d_C*(1 - zt[timeStep])*delta[timeStep];
        delta_rw += preActivationValue[timeStep]*d_rt*h_prev*hu*d_C*(1 - zt[timeStep])*delta[timeStep];
        delta_zw += preActivationValue[timeStep]*d_zt*(h_prev - C[timeStep])*delta[timeStep];
        
        //Update the previous output weights
        delta_hu += rt[timeStep]*h_prev*d_C*(1 - zt[timeStep])*delta[timeStep];
        delta_ru += h_prev*d_rt*h_prev*hu*d_C*(1 - zt[timeStep])*delta[timeStep];
        delta_zu += h_prev*d_zt*(h_prev - C[timeStep])*delta[timeStep];
        
        //Update the biases
        delta_hb += d_C*(1 - zt[timeStep])*delta[timeStep];
        delta_rb += d_rt*h_prev*hu*d_C*(1 - zt[timeStep])*delta[timeStep];
        delta_zb += d_zt*(h_prev - C[timeStep])*delta[timeStep];
        
        //Add to the delta of the previous timestep
        if(timeStep != 0){
            delta[timeStep - 1] += ru*d_rt*h_prev*hu*d_C*(1 - zt[timeStep])*delta[timeStep]
                                + hu*rt[timeStep]*d_C*(1 - zt[timeStep])*delta[timeStep]
                                + zu*d_zt*(h_prev - C[timeStep])*delta[timeStep]
                                + zt[timeStep]*delta[timeStep];
        }
        
        //Set the outgoing delta for this node
        delta[timeStep] = hw*d_C*(1 - zt[timeStep])*delta[timeStep]
                        + rw*d_rt*h_prev*hu*d_C*(1 - zt[timeStep])*delta[timeStep]
                        + zw*d_zt*(h_prev - C[timeStep])*delta[timeStep];
        
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
        
        hw = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        rw = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        zw = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        hu = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        ru = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        zu = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
        hb = Math.sqrt(2.0/fanIn)*weightGen.nextGaussian();
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
        
        hw = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        rw = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        zw = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        hu = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        ru = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        zu = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
        hb = Math.sqrt(6.0/(fanIn + fanOut))*(2.0*weightGen.nextDouble() - 1);
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
        return "[GRU Node - layer: " + layer + ", number: " + number + ", type: " + nodeType + "]";
    }
}

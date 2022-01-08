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


public class PA14Tests {

    public static void main(String[] arguments) {
        if (arguments.length != 1) {
            System.err.println("Invalid arguments, you must specify a loss function, usage: ");
            System.err.println("\tjava PA14Tests <loss function>");
            System.err.println("\tloss function options are: 'svm' or 'softmax'");
            System.exit(1);
        }

        testNormalize();

        String lossFunctionName = arguments[0];

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("svm")) {
            Log.info("Using an SVM loss function.");
            lossFunction = LossFunction.SVM;

            //test the numeric gradient calculations on tiny,
            //small and large neural networks with L2_NORM
            //loss functions
            testTinyGradientNumericSVM();
            testSmallGradientNumericSVM();
            testLargeGradientNumericSVM();

        } else if (lossFunctionName.equals("softmax")) {
            Log.info("Using an SOFTMAX loss function.");
            lossFunction = LossFunction.SOFTMAX;

            //test the numeric gradient calculations on tiny,
            //small and large neural networks with L2_NORM
            //loss functions
            testTinyGradientNumericSOFTMAX();
            testSmallGradientNumericSOFTMAX();
            testLargeGradientNumericSOFTMAX();

        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        DataSet irisData = new DataSet("iris data", "./datasets/iris.txt");

        //these tests calculation of of the gradient via
        //the backwards pass for the tiny, small and large fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights when the network has a L2_NORM
        //loss function
        PA12Tests.testTinyGradients(irisData, lossFunction);
        PA12Tests.testSmallGradients(irisData, lossFunction);
        PA12Tests.testLargeGradients(irisData, lossFunction);

        //this tests calculation of of the gradient via
        //the backwards pass for the tiny, small and large fully connected
        //neural network by comparing it to the
        //numeric gradient multiple times with random
        //starting weights when the network has a L2_NORM
        //loss function
        PA12Tests.testTinyGradientsMultiInstance(irisData, lossFunction);
        PA12Tests.testSmallGradientsMultiInstance(irisData, lossFunction);
        PA12Tests.testLargeGradientsMultiInstance(irisData, lossFunction);
     }

    public static void testNormalize() {
        DataSet dataSet = new DataSet("iris data", "./datasets/iris.txt");
        double[] means = dataSet.getInputMeans();
        double[] stdDevs = dataSet.getInputStandardDeviations();
        Log.info("calculated data set means: " + Arrays.toString(means));
        Log.info("calculated data set standard deviations: " + Arrays.toString(stdDevs));
        dataSet.normalize(means, stdDevs);

        double[] actualMeans = new double[]{5.843333333333335, 3.0540000000000007, 3.7586666666666693, 1.1986666666666672};
        double[] actualStdDevs = new double[]{0.8280661279778629, 0.4335943113621737, 1.7644204199522617, 0.7631607417008414};

        if (BasicTests.vectorsCloseEnough(means, actualMeans)) {
            Log.info("passed mean calculation from iris data set!");
        } else {
            Log.info("failed calculation from iris data set!");
        }

        if (BasicTests.vectorsCloseEnough(stdDevs, actualStdDevs)) {
            Log.info("passed standard deviation calculation from iris data set!");
        } else {
            Log.info("failed standard deviation calculation from iris data set!");
        }
     }

    /**
     * This tests calculation of the numeric gradient for
     * the tiny fully connected neural network generated
     * by PA11Tests.createTinyNeuralNetwork()
     */
    public static void testTinyGradientNumericSVM() {
        try {
            //creating this data set should work correctly if the previous test passed.
            DataSet irisData = new DataSet("iris data", "./datasets/iris.txt");

            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(irisData, LossFunction.SVM);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[tinyNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            tinyNN.setWeights(weights);
            
            double[] calculatedGradient = new double[]{0.45320735941878354, 0.14778645085478104, 0.3273164272954432, 0.10673465666855009, 0.14267639181220204, 0.04652536578220179, 0.033570914936831286, 0.010947144302164702, 0.08392728845230124, 0.0, 0.0, 0.0, 0.027367863530969316, 0.4095965278061442, -0.22239973085369513, -0.19937602013797573};
            double[] numericGradient = tinyNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 5!");
            }
            Log.info("passed testTinyGradientNumeric on instance5");


            calculatedGradient = new double[]{-1.2672578009187419, -0.0720406267973317, -0.6225126059078434, -0.03538837667349526, -1.0004666872731605, -0.05687417647948223, -0.28902370940997457, -0.0164303171068525, -0.22232593099857922, 0.0, 0.0, 0.0, -0.012638707858059206, -0.22164409418934383, 0.4897482286381205, -0.21459567012271918};
            numericGradient = tinyNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 55!");
            }
            Log.info("passed testTinyGradientNumeric on instance55");


            calculatedGradient = new double[]{0.8743167068114843, 0.026152767729215043, 0.37470716085508116, 0.011208328709599868, 0.6800241059146117, 0.0203410410737348, 0.24980477464353612, 0.007472219509807587, 0.13878042826043213, 0.0, 0.0, 0.0, 0.004151232690929874, -0.2232711215910399, -0.24711701795965269, 0.43209459565929365};
            numericGradient = tinyNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 123!");
            }
            Log.info("passed testTinyGradientNumeric on instance123");

        } catch (Exception e) {
            Log.fatal("Failed testTinyGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }



    /**
     * This tests calculation of the numeric gradient for
     * the small fully connected neural network generated
     * by PA11Tests.createSmallNeuralNetwork()
     */
    public static void testSmallGradientNumericSVM() {
        try {
            //creating this data set should work correctly if the previous test passed.
            DataSet irisData = new DataSet("iris data", "./datasets/iris.txt");

            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(irisData, LossFunction.SVM);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[smallNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            smallNN.setWeights(weights);

            double[] calculatedGradient = new double[]{-0.026934428021263557, -0.030305240539263423, -0.022280446376754526, -0.019452643940098824, -0.021887118784036375, -0.016091434851261965, -0.008479357216373273, -0.009540541690000737, -0.0070142158747898975, -0.0019951418295249823, -0.0022448332082092293, -0.0016504042577025757, -0.004987856794258505, -0.004187212798711926, -0.05574438466382503, -0.003689333283318774, -0.0056120819103000485, -0.004543589948724502, -0.06048881617815027, -0.004003333220481409, -0.00412600842381039, -0.005410265568883688, -0.0720268356069198, -0.004766957939494887, 0.006501759131083418, -0.1036369812190685, 0.23876312482684625, 0.0436169522721741, 0.0865579052877763, 0.0, 0.0, 0.0, 0.005728666430115936, -0.10533218075536865, 0.24266859632859905, 0.04433039935136662};
            double[] numericGradient = smallNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 5!");
            }
            Log.info("passed testSmallGradientNumeric on instance5");


            calculatedGradient = new double[]{0.016078822717702224, 0.004592121127799942, 0.012800138726731802, 0.007898367515579707, 0.0022557788970090087, 0.006287788067993461, 0.012693806006325303, 0.003625358901970799, 0.010105373204893908, 0.003667099957027631, 0.0010473255596110675, 0.0029193303330288245, 0.0028208457791834007, 0.0041739411926755565, 0.10852731602284393, 0.003274218673965379, 8.056344480422695E-4, 0.005206886033448654, 0.13538505760912756, 0.004084504956480828, 0.002245639230125107, 0.00485639528591264, 0.1262718885008951, 0.003809563775547531, -0.005465521368819282, 0.050538209173112136, -0.488774778428791, 0.04234775863309892, -0.14210986054585817, 0.0, 0.0, 0.0, -0.004287389332446878, 0.051006356915905826, -0.49330240559086747, 0.04274003484461275};
            numericGradient = smallNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 55!");
            }
            Log.info("passed testSmallGradientNumeric on instance55");


            calculatedGradient = new double[]{-0.004229798733490497, -0.001221918122240595, -0.0029936431111821094, -0.0018127699341619063, -5.236810984854401E-4, -0.0012829892703791757, -0.0032898439528139534, -9.503819953238235E-4, -0.0023283885930425186, -0.0012085155098873201, -3.491185118775775E-4, -8.553246999554176E-4, -6.713984923578664E-4, -0.00157110324749965, -0.04920362339788653, -0.0011607781402744877, -1.9395596240201485E-4, -0.001810869232343748, -0.05671260572270853, -0.001337927546529727, -4.751798954316655E-4, -0.001744888677990275, -0.05464628083373668, -0.0012891798739644855, 0.001874436161841686, 0.050311985688722416, 0.2454341796465087, -0.08424753339397739, 0.058703415461991426, 0.0, 0.0, 0.0, 0.0013848922009174203, 0.05070424080599878, 0.24734771120193955, -0.08490437242159032};
            numericGradient = smallNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 123!");
            }
            Log.info("passed testSmallGradientNumeric on instance123");

        } catch (Exception e) {
            Log.fatal("Failed testSmallGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    /**
     * This tests calculation of the numeric gradient for
     * the large fully connected neural network generated
     * by PA11Tests.createLargeNeuralNetwork()
     */
    public static void testLargeGradientNumericSVM() {
        try {
            //creating this data set should work correctly if the previous test passed.
            DataSet irisData = new DataSet("iris data", "./datasets/iris.txt");

            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(irisData, LossFunction.SVM);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[largeNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            largeNN.setWeights(weights);

            double[] calculatedGradient = new double[]{-3.108624468950438E-8, -3.9968028886505635E-8, -3.108624468950438E-8, -2.220446049250313E-8, -2.886579864025407E-8, -2.220446049250313E-8, -1.1102230246251565E-8, -1.3322676295501878E-8, -1.1102230246251565E-8, -4.440892098500626E-9, -4.440892098500626E-9, -4.440892098500626E-9, -4.440892098500626E-9, -4.440892098500626E-9, -0.06957379117267237, -4.440892098500626E-9, -0.033523088749376484, -4.440892098500626E-9, -6.661338147750939E-9, -4.440892098500626E-9, -0.07549525005146052, -4.440892098500626E-9, -0.03637625756169882, -4.440892098500626E-9, -4.440892098500626E-9, -4.440892098500626E-9, -0.08989569533213171, -4.440892098500626E-9, -0.04331489389386434, -4.440892098500626E-9, 4.440892098500626E-9, -4.440892098500626E-8, -0.18943286939077097, -1.3322676295501878E-8, -0.29330128770155284, 0.10803172134643546, 4.440892098500626E-8, 0.18351321129372877, 1.3322676295501878E-8, 0.28413580155373097, 4.440892098500626E-9, -4.6629367034256575E-8, -0.19135808937420506, -1.3322676295501878E-8, -0.29628212772081497, 0.05205346598202709, 4.440892098500626E-8, 0.19065792944417126, 1.3322676295501878E-8, 0.29519806155065, 4.440892098500626E-9, -4.6629367034256575E-8, -0.19261454209384965, -1.3322676295501878E-8, -0.2982275071339302, -4.884981308350689E-8, 0.49999997697725007, -0.17509119487613134, -0.24999998959884806, -0.19494155178989558, -0.3984219354435936, 0.1395203508280929, 0.1992109677217968, -1.3322676295501878E-8, 0.49999999474081847, -0.1750911970965774, -0.24999999848063226, -0.3018304495228108, -0.353671525399335, 0.12384954173327856, 0.17683576158944447};
            double[] numericGradient = largeNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 5!");
            }
            Log.info("passed testLargeGradientNumeric on instance5");


            calculatedGradient = new double[]{1.6653345369377348E-8, 4.440892098500626E-9, 1.6653345369377348E-8, 7.771561172376096E-9, 2.220446049250313E-9, 7.771561172376096E-9, 1.3322676295501878E-8, 3.3306690738754696E-9, 1.3322676295501878E-8, 3.3306690738754696E-9, 0.0, 3.3306690738754696E-9, 3.3306690738754696E-9, 3.3306690738754696E-9, 0.1650046299594976, 2.220446049250313E-9, 0.0795050114632545, 1.1102230246251565E-9, 0.0, 3.3306690738754696E-9, 0.20583906801263652, 3.3306690738754696E-9, 0.0991804727235035, 2.220446049250313E-9, 2.220446049250313E-9, 3.3306690738754696E-9, 0.19198342693371728, 2.220446049250313E-9, 0.09250433863350338, 1.1102230246251565E-9, -3.3306690738754696E-9, 8.104628079763643E-8, 0.3847885099439452, 2.3314683517128287E-8, 0.59577285993484, -0.21606343603153277, -7.771561172376096E-8, -0.36702641814656545, -1.9984014443252818E-8, -0.5682715864541166, -3.3306690738754696E-9, 8.104628079763643E-8, 0.3868491915692829, 2.3314683517128287E-8, 0.5989634344061301, -0.10410692752316208, -8.104628079763643E-8, -0.38131585666789647, -2.3314683517128287E-8, -0.5903961031172855, -2.220446049250313E-9, 8.104628079763643E-8, 0.388078301716277, 2.3314683517128287E-8, 0.6008664810153164, 8.104628079763643E-8, -0.24999998959884806, 0.3501823819807015, -0.24999998959884806, 0.38988310246956814, 0.19921096994224285, -0.2790406983255167, 0.19921096883201983, 2.3314683517128287E-8, -0.2499999962601862, 0.35018239308293175, -0.2499999962601862, 0.6036608768411611, 0.17683576158944447, -0.2476990834665571, 0.1768357626996675};
            numericGradient = largeNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 55!");
            }
            Log.info("passed testLargeGradientNumeric on instance55");


            calculatedGradient = new double[]{-4.440892098500626E-9, -4.440892098500626E-9, -4.440892098500626E-9, -4.440892098500626E-9, -2.220446049250313E-9, -4.440892098500626E-9, -4.440892098500626E-9, -2.220446049250313E-9, -4.440892098500626E-9, -2.220446049250313E-9, 0.0, -2.220446049250313E-9, -2.220446049250313E-9, -2.220446049250313E-9, -0.0905492680836062, -2.220446049250313E-9, -0.04362981087524531, -2.220446049250313E-9, 0.0, -4.440892098500626E-9, -0.10436802089230923, -2.220446049250313E-9, -0.05028816696395211, -2.220446049250313E-9, -2.220446049250313E-9, -4.440892098500626E-9, -0.10056537158575907, -2.220446049250313E-9, -0.04845591705660013, -2.220446049250313E-9, 4.440892098500626E-9, -3.9968028886505635E-8, -0.1929111848042453, -1.3322676295501878E-8, -0.29868675310851245, 0.10803170802375917, 3.9968028886505635E-8, 0.18351320907328272, 1.3322676295501878E-8, 0.2841357527039179, 2.220446049250313E-9, -3.9968028886505635E-8, -0.1937640647931005, -1.3322676295501878E-8, -0.30000727679890815, 0.05205346154113499, 3.9968028886505635E-8, 0.19065792944417126, 1.3322676295501878E-8, 0.2951980082599448, 2.220446049250313E-9, -3.9968028886505635E-8, -0.19425930863903318, -1.3322676295501878E-8, -0.300774074535326, -3.9968028886505635E-8, -0.24999998959884806, -0.17509119043523924, 0.4999999836385882, -0.19494155178989558, 0.19921096994224285, 0.1395203508280929, -0.3984219398844857, -1.3322676295501878E-8, -0.24999999848063226, -0.17509119487613134, 0.4999999925203724, -0.3018303962321056, 0.17683575936899842, 0.12384954173327856, -0.35367152317888895};
            numericGradient = largeNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 123!");
            }
            Log.info("passed testLargeGradientNumeric on instance123");


        } catch (Exception e) {
            Log.fatal("Failed testLargeGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    /**
     * This tests calculation of the numeric gradient for
     * the tiny fully connected neural network generated
     * by PA11Tests.createTinyNeuralNetwork()
     */
    public static void testTinyGradientNumericSOFTMAX() {
        try {
            //creating this data set should work correctly if the previous test passed.
            DataSet irisData = new DataSet("iris data", "./datasets/iris.txt");

            NeuralNetwork tinyNN = PA11Tests.createTinyNeuralNetwork(irisData, LossFunction.SOFTMAX);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[tinyNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            tinyNN.setWeights(weights);
            
            double[] calculatedGradient = new double[]{0.1307049390941728, 0.04334853365151048, 0.09439801162969275, 0.03130727455058491, 0.04114785068232152, 0.013646760388397183, 0.009681847545905953, 0.0032110025749432225, 0.02420461830965337, 0.0, 0.0, 0.0, 0.008027506437358056, 0.1339142408074423, -0.06687557840834302, -0.07041637273275114};
            double[] numericGradient = tinyNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 5!");
            }
            Log.info("passed testTinyGradientNumeric on instance5");

            calculatedGradient = new double[]{-0.4453843005514102, -0.025311146512052574, -0.21878526990093405, -0.012433547524892674, -0.35161918598980435, -0.019982484555924884, -0.10157887531114795, -0.005772716749063989, -0.07813759639319073, 0.0, 0.0, 0.0, -0.004440552370255091, -0.07697769310865965, 0.17196700841104473, -0.07617370290091685};
            numericGradient = tinyNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 55!");
            }
            Log.info("passed testTinyGradientNumeric on instance55");

            calculatedGradient = new double[]{0.26590430080020155, 0.007902422050065638, 0.11395898558141937, 0.0033867519899644094, 0.20681445556114397, 0.006146330111533871, 0.07597265705427958, 0.00225783391982759, 0.04220703120338953, 0.0, 0.0, 0.0, 0.0012543521776819944, -0.07756895015198495, -0.07350276720607951, 0.1393206028321714};
            numericGradient = tinyNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testTinyGradientNumeric, instance 123!");
            }
            Log.info("passed testTinyGradientNumeric on instance123");

        } catch (Exception e) {
            Log.fatal("Failed testTinyGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }



    /**
     * This tests calculation of the numeric gradient for
     * the small fully connected neural network generated
     * by PA11Tests.createSmallNeuralNetwork()
     */
    public static void testSmallGradientNumericSOFTMAX() {
        try {
            //creating this data set should work correctly if the previous test passed.
            DataSet irisData = new DataSet("iris data", "./datasets/iris.txt");

            NeuralNetwork smallNN = PA11Tests.createSmallNeuralNetwork(irisData, LossFunction.SOFTMAX);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[smallNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            smallNN.setWeights(weights);

            double[] calculatedGradient = new double[]{-0.011431938728989621, -0.012863152765163477, -0.009457247207578234, -0.008256401118345025, -0.009290055391630858, -0.00683023304581809, -0.0035989433655458924, -0.0040495118369676675, -0.0029772817544682084, -8.468103995795673E-4, -9.528267064240481E-4, -7.005362956391536E-4, -0.0021170232233913566, -0.0017817569641920272, -0.023651685054915106, -0.0015619383564313694, -0.002382064545614071, -0.001933403437348602, -0.025664689262683282, -0.0016948764613999856, -0.0017513412942093964, -0.0023021917705534634, -0.030560137753710137, -0.0020181689652787327, 0.002766649132013299, -0.03720908803295231, 0.10477535061781396, 0.012179611763585285, 0.03672549708788608, 0.0, 0.0, 0.0, 0.002425325495991615, -0.03781772006661299, 0.1064891697044601, 0.012378834624016122};
            double[] numericGradient = smallNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 5!");
            }
            Log.info("passed testSmallGradientNumeric on instance5");


            calculatedGradient = new double[]{0.004507229034445004, 0.0012872675148045687, 0.0035881436710738512, 0.002214077254869551, 6.323414014630657E-4, 0.001762596180121534, 0.0035583375135317397, 0.001016263739828105, 0.002832744594449821, 0.0010279649353961418, 2.935873766318764E-4, 8.183476118972521E-4, 7.907413612784353E-4, 0.0011700196367314675, 0.030422584451628154, 9.178519055907941E-4, 2.2583657166563853E-4, 0.0014595696873342945, 0.03795139813522752, 0.001144996319979441, 6.29498675408513E-4, 0.0013613205007700913, 0.03539677606578806, 0.001067923527386938, -0.0015320689161768541, 0.014229273315180535, -0.13697920631106086, 0.011812719136194971, -0.03983650742611644, 0.0, 0.0, 0.0, -0.0012018708250849386, 0.014361081213110083, -0.13824807409967832, 0.011922143827725051};
            numericGradient = smallNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 55!");
            }
            Log.info("passed testSmallGradientNumeric on instance55");


            calculatedGradient = new double[]{-0.001946692806953365, -5.623279619726418E-4, -0.0013776135787679777, -8.342970758690171E-4, -2.4099833240143198E-4, -5.904066124884366E-4, -0.0015140944054081729, -4.3736569921293267E-4, -0.0010714784615117878, -5.561984206536863E-4, -1.606648147856049E-4, -3.936051484743075E-4, -3.0899727221367357E-4, -7.201150786784183E-4, -0.02266501542003141, -5.367928324062632E-4, -8.925749028776409E-5, -8.30011614993964E-4, -0.02612392857592738, -6.187128587242796E-4, -2.1866952693017083E-4, -7.997691398031748E-4, -0.025172103290671544, -5.961720006553151E-4, 8.591483080522266E-4, 0.014162221395608299, 0.1078972755585994, -0.030375699733298234, 0.02704097279426776, 0.0, 0.0, 0.0, 6.404332619780462E-4, 0.014272636406076344, 0.10873849709547301, -0.030612523627127075};
            numericGradient = smallNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testSmallGradientNumeric, instance 123!");
            }
            Log.info("passed testSmallGradientNumeric on instance123");

        } catch (Exception e) {
            Log.fatal("Failed testSmallGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }


    /**
     * This tests calculation of the numeric gradient for
     * the large fully connected neural network generated
     * by PA11Tests.createLargeNeuralNetwork()
     */
    public static void testLargeGradientNumericSOFTMAX() {
        try {
            //creating this data set should work correctly if the previous test passed.
            DataSet irisData = new DataSet("iris data", "./datasets/iris.txt");

            NeuralNetwork largeNN = PA11Tests.createLargeNeuralNetwork(irisData, LossFunction.SOFTMAX);

            //get an instance from each iris class
            Instance instance5 = irisData.getInstance(5);
            Instance instance55 = irisData.getInstance(55);
            Instance instance123 = irisData.getInstance(123);

            double[] weights = new double[largeNN.getNumberWeights()];

            for (int i = 0; i < weights.length; i++) {
                //give the test weights a spread of positive and negative values
                weights[i] = (-1 * (i%2)) * 0.05 * i;
            }

            largeNN.setWeights(weights);

            double[] calculatedGradient = new double[]{-1.2212453270876722E-8, -1.6653345369377348E-8, -1.1102230246251565E-8, -9.992007221626409E-9, -9.992007221626409E-9, -7.771561172376096E-9, -4.440892098500626E-9, -5.551115123125783E-9, -4.440892098500626E-9, -1.1102230246251565E-9, -1.1102230246251565E-9, -1.1102230246251565E-9, -3.3306690738754696E-9, -1.1102230246251565E-9, -0.02759628858761687, -1.1102230246251565E-9, -0.01329685805906422, -1.1102230246251565E-9, -3.3306690738754696E-9, -1.1102230246251565E-9, -0.029945023127808668, -1.1102230246251565E-9, -0.0144285605685468, -1.1102230246251565E-9, -3.3306690738754696E-9, -1.1102230246251565E-9, -0.03565692741069881, -1.1102230246251565E-9, -0.01718075681722553, -1.1102230246251565E-9, 3.3306690738754696E-9, -1.887379141862766E-8, -0.07513812683157539, -6.661338147750939E-9, -0.11633730312432533, 0.04285054200359184, 1.7763568394002505E-8, 0.07279010727501145, 6.661338147750939E-9, 0.1127018345314923, 1.1102230246251565E-9, -1.9984014443252818E-8, -0.07590176043237307, -6.661338147750939E-9, -0.11751964734685316, 0.020646891973896686, 1.887379141862766E-8, 0.07562404480410123, 6.661338147750939E-9, 0.11708965463874677, 1.1102230246251565E-9, -1.9984014443252818E-8, -0.07640012955612008, -6.661338147750939E-9, -0.11829127677387419, -2.1094237467877974E-8, 0.17458096412958923, -0.06944953279131028, -0.0754190243590358, -0.07732313456187967, -0.13911377827469096, 0.05534043778787634, 0.06009719055732887, -6.661338147750939E-9, 0.17458097079092738, -0.06944953501175632, -0.07541902546925883, -0.11972037694540916, -0.12348863598354853, 0.04912464546258377, 0.05334712560589594};
            double[] numericGradient = largeNN.getNumericGradient(instance5);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 5!");
            }
            Log.info("passed testLargeGradientNumeric on instance5");


            calculatedGradient = new double[]{4.9960036108132044E-9, 5.551115123125783E-10, 4.9960036108132044E-9, 2.220446049250313E-9, 5.551115123125783E-10, 2.220446049250313E-9, 4.440892098500626E-9, 5.551115123125783E-10, 4.440892098500626E-9, 5.551115123125783E-10, 0.0, 5.551115123125783E-10, 5.551115123125783E-10, 5.551115123125783E-10, 0.04977795564631293, 5.551115123125783E-10, 0.023984762509421387, 5.551115123125783E-10, 0.0, 5.551115123125783E-10, 0.06209672887003137, 5.551115123125783E-10, 0.02992037895221955, 5.551115123125783E-10, 5.551115123125783E-10, 5.551115123125783E-10, 0.057916813567260306, 5.551115123125783E-10, 0.02790634889837662, 5.551115123125783E-10, -5.551115123125783E-10, 2.4980018054066022E-8, 0.11608150052300203, 6.661338147750939E-9, 0.17973043886509998, -0.06518117656728606, -2.4980018054066022E-8, -0.11072310124315976, -6.106226635438361E-9, -0.17143396091601204, -5.551115123125783E-10, 2.4980018054066022E-8, 0.11670315880341064, 6.661338147750939E-9, 0.18069295892875203, -0.03140657345301889, -2.4980018054066022E-8, -0.11503388408495852, -6.661338147750939E-9, -0.17810839969545356, -5.551115123125783E-10, 2.4980018054066022E-8, 0.11707395219939798, 6.661338147750939E-9, 0.18126706247123536, 2.4980018054066022E-8, -0.07541902491414731, 0.10564165764392897, -0.07541902380392429, 0.11761841722801591, 0.06009719055732887, -0.08417991192999352, 0.06009719000221736, 6.661338147750939E-9, -0.07541902602437034, 0.10564166208482106, -0.07541902491414731, 0.18211006425072895, 0.05334712560589594, -0.07472489571558327, 0.0533471239405614};
            numericGradient = largeNN.getNumericGradient(instance55);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 55!");
            }
            Log.info("passed testLargeGradientNumeric on instance55");


            calculatedGradient = new double[]{-2.220446049250313E-9, 0.0, -2.220446049250313E-9, 0.0, 0.0, 0.0, -2.220446049250313E-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.035916166707394837, 0.0, -0.01730566689950308, 0.0, 0.0, 0.0, -0.04139734555153041, 0.0, -0.019946689855387945, 0.0, 0.0, 0.0, -0.03988903207741146, 0.0, -0.01921993342257622, 0.0, 0.0, -1.554312234475219E-8, -0.07651779432293893, -4.440892098500626E-9, -0.11847344105753166, 0.04285053756269974, 1.554312234475219E-8, 0.07279010727501145, 4.440892098500626E-9, 0.11270181787814693, 0.0, -1.554312234475219E-8, -0.07685608705010338, -5.551115123125783E-9, -0.11899722318631234, 0.02064689086367366, 1.554312234475219E-8, 0.07562404480410123, 4.440892098500626E-9, 0.11708963687517837, 0.0, -1.554312234475219E-8, -0.07705252436096544, -5.551115123125783E-9, -0.11930137100435445, -1.554312234475219E-8, -0.07541902546925883, -0.06944953279131028, 0.1745809674602583, -0.07732313345165664, 0.06009719055732887, 0.055340436677653315, -0.13911378160536003, -5.551115123125783E-9, -0.07541902657948185, -0.0694495339015333, 0.1745809719011504, -0.11972036029206379, 0.053347124495672915, 0.04912464546258377, -0.12348863931421761};
            numericGradient = largeNN.getNumericGradient(instance123);
            Log.info("numericGradient: " + Arrays.toString(numericGradient));
            Log.info("calculatedGradient: " + Arrays.toString(calculatedGradient));

            if (!BasicTests.gradientsCloseEnough(calculatedGradient, numericGradient)) {
                throw new NeuralNetworkException("Gradients not close enough on testLargeGradientNumeric, instance 123!");
            }
            Log.info("passed testLargeGradientNumeric on instance123");


        } catch (Exception e) {
            Log.fatal("Failed testLargeGradientNumeric");
            Log.fatal("Threw exception: " + e);
            e.printStackTrace();
        }
    }

}


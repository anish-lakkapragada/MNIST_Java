package MNIST_DNN.dev_files;


import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MNIST10k {

    public static double beta1 = 0.9;
    public static double beta2 = 0.999;
    public static double biasBeta1 = beta1;
    public static double biasBeta2 = beta2;
    public static boolean adam = false;

    public static double alpha = 1; //ELU function alpha
    public static ArrayList<ArrayList<Double>> x_train = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> y_train = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> x_test = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> y_test = new ArrayList<ArrayList<Double>>();
    public static int numData = 20000;
    public static int numVal = 1000;
    public static int miniBatchSize = 20000;
    public static int numNeurons1 = 128;
    public static int numNeurons2 = 64;
    public static int numNeurons3 = 10;
    public static double learningRate = 0.000000005;
    public static double biasLearningRate = 0.0002;
    public static double momentum = 0;
    public static double biasMomentum = 0;
    public static double gradClip = 100000;
    public static boolean shouldGradClip = true;
    public static int numEpochs = 200000;
    public static double e = 0.01;
    public static boolean useNesterov = false;
    public static ArrayList<ArrayList<Double>> weights1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> weights2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> weights3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> velocity1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> velocity2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> velocity3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> biasVelocity1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> biasVelocity2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> biasVelocity3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamM1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamM2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamM3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamV1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamV2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamV3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamBM1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamBM2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamBM3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamBV1 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamBV2 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> adamBV3 = new ArrayList<ArrayList<Double>>();
    public static ArrayList<Double> biases1 = new ArrayList<Double>();
    public static ArrayList<Double> biases2 = new ArrayList<Double>();
    public static ArrayList<Double> biases3 = new ArrayList<Double>();
    public static ArrayList<ArrayList<Double>> pastCorrects = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> pastLosses = new ArrayList<ArrayList<Double>>();
    public static void getTrainData() throws Exception {
        String train_path = "/Users/anish/Java Fun/ML Java/src/mnist_train.csv";
        String line = "";
        BufferedReader br = new BufferedReader(new FileReader(train_path));
        int z = numData;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            ArrayList<Double> currentX_train = new ArrayList<Double>();
            for (int i = 0; i < values.length; i++) {
                if (i == 0) {y_train.add(oneHot(Double.parseDouble(values[0])));}
                else {
                    currentX_train.add((Double.parseDouble(values[i]))/255.0);
                }
            }
            x_train.add(currentX_train);
            z -= 1;
            if (z == 0) {break;}
        }
    }

    public static void getValidationData() throws Exception {
        String train_path = "/Users/anish/Java Fun/ML Java/src/mnist_test.csv";
        String line = "";
        BufferedReader br = new BufferedReader(new FileReader(train_path));
        int z = numVal;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            ArrayList<Double> currentX_train = new ArrayList<Double>();
            for (int i = 0; i < values.length; i++) {
                if (i == 0) {y_test.add(oneHot(Double.parseDouble(values[0])));}
                else {
                    currentX_train.add((Double.parseDouble(values[i]))/255.0);
                }
            }
            x_test.add(currentX_train);
            z -= 1;
            if (z == 0) {break;}
        }
    }

    public static void biasMomentumInit() {
        ArrayList<Double> sample1 = new ArrayList<Double>();
        ArrayList<Double> sample2 = new ArrayList<Double>();
        ArrayList<Double> sample3 = new ArrayList<Double>();

        for (int n1 = 0; n1 < numNeurons1; n1++) {sample1.add(0.0);}
        for (int n2 = 0; n2 < numNeurons2; n2++) {sample2.add(0.0);}
        for (int n3 = 0; n3 <  numNeurons3; n3++) {sample3.add(0.0);}

        for (int data = 0; data < miniBatchSize; data++) {
            biasVelocity1.add(sample1);
            biasVelocity2.add(sample2);
            biasVelocity3.add(sample3);
        }
    }

    public static ArrayList<Double> oneHot(double x) {
        ArrayList<Double> oneHotted = new ArrayList<Double>();
        for (int deep = 0; deep < 10; deep++) {
            if (deep == x) {oneHotted.add(1.0);}
            else {oneHotted.add(0.0);}
        }
        return oneHotted;
    }

    public static void paramInit() {
        Random rnd = new Random();
        for (int feature = 0; feature < 784; feature++) {
            ArrayList<Double> currentWeight = new ArrayList<Double>();
            ArrayList<Double> currentVeloc = new ArrayList<Double>();
            for (int neuron =0; neuron < numNeurons1; neuron++) {
                if (feature == 0) {biases1.add(rnd.nextDouble());}
                currentWeight.add(rnd.nextGaussian());
                currentVeloc.add(0.0);
            }
            velocity1.add(currentVeloc);
            weights1.add(currentWeight);
        }

        rnd = new Random(2190);
        for (int feature = 0; feature < numNeurons1; feature++) {
            ArrayList<Double> currentWeight = new ArrayList<Double>();
            ArrayList<Double> currentVeloc = new ArrayList<Double>();
            for (int neuron = 0; neuron < numNeurons2; neuron++) {
                if (feature == 0) {biases2.add(rnd.nextDouble());}
                currentWeight.add(rnd.nextGaussian());
                currentVeloc.add(0.0);
            }
            velocity2.add(currentVeloc);
            weights2.add(currentWeight);
        }

        rnd = new Random(10);
        for (int feature = 0; feature < numNeurons2; feature++) {
            ArrayList<Double> currentWeight = new ArrayList<Double>();
            ArrayList<Double> currentVeloc = new ArrayList<Double>();
            for (int neuron = 0; neuron < numNeurons3; neuron++) {
                if(feature==0) {biases3.add(rnd.nextDouble());}
                currentWeight.add(rnd.nextGaussian());
                currentVeloc.add(0.0);
            }
            velocity3.add(currentVeloc);
            weights3.add(currentWeight);
        }
    }

    public static void XavierInit() {
        for (int feature = 0; feature < weights1.size(); feature++) {
            for (int neuron = 0; neuron < weights1.get(feature).size(); neuron++) {
                weights1.get(feature).set(neuron, weights1.get(feature).get(neuron) * Math.sqrt(2.0/784.0));
            }
        }

        for (int feature = 0; feature < weights2.size(); feature++) {
            for (int neuron = 0; neuron < weights2.get(feature).size(); neuron++) {
                weights2.get(feature).set(neuron, weights2.get(feature).get(neuron) * Math.sqrt(2.0/weights2.get(0).size()));
            }
        }

        for (int feature = 0; feature < weights3.size(); feature++) {
            for (int neuron = 0; neuron < weights3.get(feature).size(); neuron++){
                weights3.get(feature).set(neuron, weights3.get(feature).get(neuron) * Math.sqrt(2.0/weights3.get(0).size()));
            }
        }

        for (int bias = 0; bias < biases1.size(); bias++) {
            biases1.set(bias, 0.0);
        }

        for (int bias = 0; bias < biases2.size(); bias++) {
            biases2.set(bias, 0.0);
        }

        for (int bias = 0; bias < biases3.size(); bias++) {
            biases3.set(bias, 0.0);
        }

    }

    public static ArrayList<ArrayList<Double>> standardizeData(ArrayList<ArrayList<Double>> data) {
        ArrayList<ArrayList<Double>> newData = new ArrayList<ArrayList<Double>>();
        for (int column = 0; column < data.get(0).size(); column++) {
            ArrayList<Double> currentData = new ArrayList<Double>();
            ArrayList<Double> currentFeature = getCol(data, column);
            double mean = meanList(currentFeature);
            double std = standardDeviation(currentFeature, mean);
            for (int feature = 0; feature < currentFeature.size(); feature++) {
                currentData.add((currentFeature.get(feature) - mean)/std);
            }
            newData.add(currentData);
        }
        newData = transpose(newData);
        return newData;
    }

    public static double standardDeviation(ArrayList<Double> data, double mean) {
        double num = 0;
        for (int i =0; i < data.size(); i++)  {
            num += Math.pow((data.get(i) - mean), 2);
        }
        return Math.sqrt(num/data.size());
    }

    public static double meanList(ArrayList<Double> x) {
        double sum =0;
        for (int i = 0; i < x.size(); i++) {sum += x.get(i);}
        return sum;
    }

    public static ArrayList<Double> vectorSquared(ArrayList<Double> x) {
        ArrayList<Double> newX = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {newX.add(x.get(i) * x.get(i));}
        return newX;
    }

    public static ArrayList<Double> vectorSqrt(ArrayList<Double> x) {
        ArrayList<Double> newX = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {newX.add(Math.sqrt(x.get(i)));}
        return newX;
    }

    public static ArrayList<ArrayList<Double>> ELU(ArrayList<ArrayList<Double>> z) {
        ArrayList<ArrayList<Double>> activation = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < z.size(); i++) {
            ArrayList<Double> currentActivation = new ArrayList<Double>();
            for (int j = 0; j < z.get(i).size(); j++) {
                double currentNeuron = z.get(i).get(j);
                if (currentNeuron < 0) {currentActivation.add(alpha * (Math.exp(currentNeuron) - 1));}
                if (currentNeuron >=0) {currentActivation.add(currentNeuron);}
            }
            activation.add(currentActivation);
        }
        return activation;
    }

    public static HashMap<String, Object> forwardPropagation(ArrayList<ArrayList<Double>> inputs, ArrayList<ArrayList<Double>> weights1, ArrayList<ArrayList<Double>> weights2, ArrayList<ArrayList<Double>> weights3) throws Exception{

        ArrayList<ArrayList<Double>> activation1 = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < inputs.size(); i++) {
            ArrayList<Double> currentActivation = new ArrayList<Double>();
            for (int column = 0; column < weights1.get(0).size(); column++) {
                currentActivation.add(dotSum(inputs.get(i), getCol(weights1, column)));
            }
            activation1.add(currentActivation);
            System.out.println("Activation 1: " + i);
        }

        activation1 = ELU(biasAdd(activation1, biases1));

        ArrayList<ArrayList<Double>> activation2 = new ArrayList<ArrayList<Double>>();
        for (int activation = 0; activation < activation1.size(); activation++) {
            ArrayList<Double> currentActivation = new ArrayList<Double>();
            for (int column = 0; column < weights2.get(0).size(); column++) {
                currentActivation.add(dotSum(activation1.get(activation), getCol(weights2, column)));
            }
            activation2.add(currentActivation);
            System.out.println("Activation 2: " + activation);
        }

        activation2 = ELU(biasAdd(activation2, biases2));
        ArrayList<ArrayList<Double>> activation3 = new ArrayList<ArrayList<Double>>();
        for (int activation = 0 ; activation < activation2.size(); activation++) {
            ArrayList<Double> currentActivation = new ArrayList<Double>();
            for (int column = 0; column < weights3.get(0).size(); column++) {
                currentActivation.add(dotSum(activation2.get(activation), getCol(weights3, column)));
            }
            activation3.add(currentActivation);
            System.out.println("Activation 3: " + activation);
        }

        ArrayList<ArrayList<Double>> softmaxActivation3 = softmaxActivation(biasAdd(activation3, biases3));

        HashMap<String, Object> allForward = new HashMap<String, Object>();
        allForward.put("rawA3", (Object) activation3);
        allForward.put("activation3", (Object) softmaxActivation3);
        allForward.put("activation2", (Object) activation2);
        allForward.put("activation1", (Object) activation1);
        return allForward;
    }

    public static HashMap<String, Object> backpropagation(ArrayList<ArrayList<Double>> y_train, ArrayList<ArrayList<Double>> outputs, ArrayList<ArrayList<Double>> rawA3, ArrayList<ArrayList<Double>> activation2, ArrayList<ArrayList<Double>> activation1) {
        if (useNesterov) {nesterov();}
        ArrayList<ArrayList<Double>> dLdYh = new ArrayList<ArrayList<Double>>();
        for (int output = 0; output < outputs.size(); output++) {
            ArrayList<Double> upward = elementWiseDiv(y_train.get(output), scalarAdd(outputs.get(output), e));
            dLdYh.add(scalarMultiply(upward, -1.0));
        }

        System.out.println("DLDYH shape : " + shape(dLdYh));

        ArrayList<ArrayList<ArrayList<Double>>> dYhdZ3 = new ArrayList<ArrayList<ArrayList<Double>>>();
        for (int raw = 0; raw < rawA3.size();raw++) {
            dYhdZ3.add(softmaxDeriv(rawA3.get(raw)));
        }

        System.out.println("dYhdZ3 shape : " + shape(dYhdZ3.get(0)) + " size : " + dYhdZ3.size());

        ArrayList<ArrayList<Double>> dLdZ3 = new ArrayList<ArrayList<Double>>();
        for (int output = 0; output < outputs.size(); output++) {
            dLdZ3.add(dot(dYhdZ3.get(output), dLdYh.get(output)));
            //dLdZ3.add(vectorSubtract(outputs.get(output), y_train.get(output)));
        }

        System.out.println("dYhdZ3 shape : " + shape(dLdZ3));

        ArrayList<ArrayList<Double>> gradients3 = new ArrayList<ArrayList<Double>>();
        for (int col = 0; col < activation2.get(0).size(); col++) {
            gradients3.add(dot(transpose(dLdZ3), getCol(activation2, col)));
        }

        System.out.println("grad3 shape : " + shape(gradients3));

        ArrayList<ArrayList<Double>> dLdA2 = new ArrayList<ArrayList<Double>>();
        /*for (int neuron = 0 ; neuron < weights3.size(); neuron++) {
            dLdA2.add(dot(weights3, dLdZ3.get(neuron)));
        }*/

        for (int deriv = 0; deriv < dLdZ3.size(); deriv++) {
            dLdA2.add(dot(weights3, dLdZ3.get(deriv)));
        }

        System.out.println("dLdA2 shape :" + shape(dLdA2));

        ArrayList<ArrayList<Double>> eLUDerivZ2 = new ArrayList<ArrayList<Double>>();
        for (int neurons = 0; neurons < activation2.size(); neurons++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int neuron = 0; neuron < activation2.get(neurons).size(); neuron++) {
                double rawNeuron = reverseELU(activation2.get(neurons).get(neuron));
                if (rawNeuron > 0) {currentDeriv.add(1.0);}
                else {currentDeriv.add(alpha * Math.exp(rawNeuron));}
            }
            eLUDerivZ2.add(currentDeriv);
        }

        System.out.println("ELU deriv z2 shape : " + shape(eLUDerivZ2));

        ArrayList<ArrayList<Double>> dLdZ2 = new ArrayList<ArrayList<Double>>();
        for (int row = 0; row < eLUDerivZ2.size(); row++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int col = 0; col < eLUDerivZ2.get(row).size(); col++) {
                currentDeriv.add(eLUDerivZ2.get(row).get(col) * dLdA2.get(row).get(col));
            }
            dLdZ2.add(currentDeriv);
        }

        System.out.println("DLdZ2 shape : " + shape(dLdZ2));

        ArrayList<ArrayList<Double>> gradients2 = new ArrayList<ArrayList<Double>>();
        for (int col = 0; col < activation1.get(0).size(); col++) {
            gradients2.add(dot(transpose(dLdZ2), getCol(activation1, col)));
        }

        System.out.println("grad2 shape : " + shape(gradients2));

        ArrayList<ArrayList<Double>> dLdA1 = new ArrayList<ArrayList<Double>>();
        /*for (int deriv = 0; neuron < weights2.size(); neuron++) {
            dLdA1.add(dot(dLdZ2, weights2.get(neuron)));
        }*/

        for (int deriv = 0; deriv < dLdZ2.size(); deriv++) {
            dLdA1.add(dot(weights2, dLdZ2.get(deriv)));
            //System.out.println("DOT(W2, deriv) : " + dot(weights2, dLdZ2.get(deriv)));
        }

        System.out.println("dLdA1 shape : " + shape(dLdA1));

        ArrayList<ArrayList<Double>> eLUDerivZ1 = new ArrayList<ArrayList<Double>>();
        for (int neurons = 0; neurons < activation1.size(); neurons++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int neuron = 0; neuron < activation1.get(neurons).size(); neuron++) {
                double rawNeuron = reverseELU(activation1.get(neurons).get(neuron));
                if (rawNeuron > 0) {currentDeriv.add(1.0);}
                else {currentDeriv.add(alpha * Math.exp(rawNeuron));}
            }
            eLUDerivZ1.add(currentDeriv);
        }

        System.out.println("Shape : " + shape(eLUDerivZ1));

        ArrayList<ArrayList<Double>> dLdZ1 = new ArrayList<ArrayList<Double>>();
        for (int row = 0; row < eLUDerivZ1.size(); row++) {
            ArrayList<Double> currentDeriv = new ArrayList<Double>();
            for (int col = 0; col < dLdA1.get(row).size(); col++) {
                currentDeriv.add(eLUDerivZ1.get(row).get(col) * dLdA1.get(row).get(col));
            }
            dLdZ1.add(currentDeriv);
        }

        System.out.println("dLdZ1 Shape : " + shape(dLdZ1));


        ArrayList<ArrayList<Double>> gradients1 = new ArrayList<ArrayList<Double>>();
        for (int col = 0; col < x_train.get(0).size(); col++) {
            gradients1.add(dot(transpose(dLdZ1), getCol(x_train, col)));
        }

        System.out.println(" grad 1 Shape : " + shape(gradients1));


        if (shouldGradClip) {
            gradients1 = gradClip(gradients1);
            gradients2 = gradClip(gradients2);
            gradients3 = gradClip(gradients3);
            dLdZ1 = gradClip(dLdZ1);
            dLdZ2 = gradClip(dLdZ2);
            dLdZ3 = gradClip(dLdZ3);
        }



        HashMap<String, Object> gradients = new HashMap<String, Object>();
        gradients.put("gradients1", (Object) gradients1);
        gradients.put("gradients2", (Object) gradients2);
        gradients.put("gradients3", (Object) gradients3);
        gradients.put("bgradients1", (Object) dLdZ1);
        gradients.put("bgradients2", (Object) dLdZ2);
        gradients.put("bgradients3", (Object) dLdZ3);
        return gradients;
    }


    public static double getLosses(ArrayList<ArrayList<Double>> yH, ArrayList<ArrayList<Double>> labels) {
        double losses = 0;
        for (int output = 0; output < yH.size(); output++) {
            losses += (getLoss(yH.get(output), labels.get(output)));
        }
        return losses;
    }

    public static double getLoss(ArrayList<Double> predicted, ArrayList<Double> label) {
        return vectorSum(vectorSquared(vectorSubtract(predicted, label)));
    }

    public static double vectorSum(ArrayList<Double> x ) {
        double sum =0;
        for (int i =0; i < x.size(); i++) {
            sum += x.get(i);
        }
        return sum;
    }



    public static void useMomentum(ArrayList<ArrayList<Double>> gradients1, ArrayList<ArrayList<Double>> gradients2, ArrayList<ArrayList<Double>> gradients3, ArrayList<ArrayList<Double>> bgrad1, ArrayList<ArrayList<Double>> bgrad2, ArrayList<ArrayList<Double>> bgrad3) {
        ArrayList<ArrayList<Double>> newVelocity1 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> newVelocity2 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> newVelocity3 = new ArrayList<ArrayList<Double>>();

        for (int feature = 0; feature < velocity1.size(); feature++) {
            ArrayList<Double> currentVeloc = new ArrayList<Double>();
            for (int neuron = 0; neuron < velocity1.get(feature).size(); neuron++) {
                currentVeloc.add(velocity1.get(feature).get(neuron) * momentum + learningRate * gradients1.get(feature).get(neuron));
            }
            newVelocity1.add(currentVeloc);
        }

        for (int feature = 0; feature < velocity2.size(); feature++){
            ArrayList<Double> currentVeloc = new ArrayList<Double>();
            for (int neuron = 0; neuron < velocity2.get(feature).size(); neuron++) {
                currentVeloc.add(velocity2.get(feature).get(neuron) * momentum + learningRate * gradients2.get(feature).get(neuron));
            }
            newVelocity2.add(currentVeloc);
        }

        for (int feature = 0; feature < velocity3.size(); feature++) {
            ArrayList<Double> currentVeloc = new ArrayList<Double>();
            for (int neuron = 0; neuron < velocity3.get(feature).size(); neuron++) {
                currentVeloc.add(velocity3.get(feature).get(neuron) * momentum + learningRate * gradients3.get(feature).get(neuron));
            }
            newVelocity3.add(currentVeloc);
        }

        velocity1 = newVelocity1;
        velocity2 = newVelocity2;
        velocity3 = newVelocity3;

        System.out.println("V1 : " + velocity1);
        System.out.println("V2 : " + velocity2);
        System.out.println("V3 : " + velocity3);

        for (int feature = 0; feature < weights1.size(); feature++) {
            for (int neuron = 0; neuron < weights1.get(feature).size(); neuron++) {
                weights1.get(feature).set(neuron, weights1.get(feature).get(neuron) - velocity1.get(feature).get(neuron));
            }
        }

        for (int feature =0 ; feature < weights2.size(); feature++) {
            for (int neuron = 0; neuron < weights2.get(feature).size(); neuron++) {
                weights2.get(feature).set(neuron, weights2.get(feature).get(neuron) - velocity2.get(feature).get(neuron));
            }
        }

        for (int feature =0 ;feature < weights3.size(); feature++) {
            for (int neuron = 0; neuron < weights3.get(feature).size(); neuron++) {
                weights3.get(feature).set(neuron, weights3.get(feature).get(neuron) - velocity3.get(feature).get(neuron));
            }
        }

        ArrayList<ArrayList<Double>> newBiasVelocity1 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> newBiasVelocity2 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> newBiasVelocity3 = new ArrayList<ArrayList<Double>>();
        for (int data = 0; data < biasVelocity1.size(); data++){
            newBiasVelocity1.add(vectorAdd(scalarMultiply(biasVelocity1.get(data), biasMomentum), scalarMultiply(bgrad1.get(data), biasLearningRate)));
        }

        for (int data = 0; data < biasVelocity2.size(); data++) {
            newBiasVelocity2.add(vectorAdd(scalarMultiply(biasVelocity2.get(data), biasMomentum), scalarMultiply(bgrad2.get(data), biasLearningRate)));
        }

        for (int data = 0; data < biasVelocity2.size(); data++) {
            newBiasVelocity3.add(vectorAdd(scalarMultiply(biasVelocity3.get(data), biasMomentum), scalarMultiply(bgrad3.get(data), biasLearningRate)));
        }

        biasVelocity1 = newBiasVelocity1;
        biasVelocity2 = newBiasVelocity2;
        biasVelocity3 = newBiasVelocity3;

        for (int bgrad = 0; bgrad < bgrad1.size(); bgrad++) {
            biases1 = vectorSubtract(biases1, biasVelocity1.get(bgrad));
        }

        for (int bgrad = 0; bgrad< bgrad2.size(); bgrad++) {
            biases2 = vectorSubtract(biases2, biasVelocity2.get(bgrad));
        }

        for (int bgrad = 0; bgrad < bgrad3.size(); bgrad++) {
            biases3 = vectorSubtract(biases3, biasVelocity3.get(bgrad));
        }

    }

    public static void adamInit() {
        for (int feature = 0 ; feature < 784; feature++) {
            ArrayList<Double> currentM = new ArrayList<Double>();
            for (int neuron = 0; neuron < numNeurons1; neuron++) {
                currentM.add(0.0);
            }
            adamM1.add(currentM);
            adamV1.add(currentM);
        }

        for (int feature = 0 ; feature < numNeurons1; feature++) {
            ArrayList<Double> currentM = new ArrayList<Double>();
            for (int neuron = 0; neuron < numNeurons2; neuron++) {
                currentM.add(0.0);
            }
            adamM2.add(currentM);
            adamV2.add(currentM);
        }

        for (int feature = 0 ; feature < numNeurons2; feature++) {
            ArrayList<Double> currentM = new ArrayList<Double>();
            for (int neuron = 0; neuron < numNeurons3; neuron++) {
                currentM.add(0.0);
            }
            adamM3.add(currentM);
            adamV3.add(currentM);
        }

        ArrayList<Double> sample1 = new ArrayList<Double>();
        ArrayList<Double> sample2 = new ArrayList<Double>();
        ArrayList<Double> sample3 = new ArrayList<Double>();

        for (int n1 = 0; n1 < numNeurons1; n1++) {sample1.add(0.0);}
        for (int n2 = 0; n2 < numNeurons2; n2++) {sample2.add(0.0);}
        for (int n3 = 0; n3 <  numNeurons3; n3++) {sample3.add(0.0);}

        for (int data = 0; data < miniBatchSize; data++) {
            adamBM1.add(sample1);
            adamBM2.add(sample2);
            adamBM3.add(sample3);
            adamBV1.add(sample1);
            adamBV2.add(sample2);
            adamBV3.add(sample3);
        }


    }


    public static ArrayList<ArrayList<Double>> matrixScalarDivide(ArrayList<ArrayList<Double>> matrix, double scalar) {
        ArrayList<ArrayList<Double>> newMatrix = new ArrayList<ArrayList<Double>>();
        for (int i = 0 ; i < matrix.size(); i++) {newMatrix.add(scalarDivide(matrix.get(i), scalar));}
        return newMatrix;
    }

    public static void adam(ArrayList<ArrayList<Double>> gradients1, ArrayList<ArrayList<Double>> gradients2, ArrayList<ArrayList<Double>> gradients3, ArrayList<ArrayList<Double>> bgrads1, ArrayList<ArrayList<Double>> bgrads2, ArrayList<ArrayList<Double>> bgrads3, double epoch, double numOutputs) {
        //scale grads
        gradients1 = matrixScalarDivide(gradients1, numOutputs);
        gradients2 = matrixScalarDivide(gradients2, numOutputs);
        gradients3 = matrixScalarDivide(gradients3, numOutputs);
        bgrads1 = matrixScalarDivide(bgrads1, numOutputs);
        bgrads2 = matrixScalarDivide(bgrads2, numOutputs);
        bgrads3 = matrixScalarDivide(bgrads3, numOutputs);

        //update weights

        //1. update adamM
        for (int feature =0; feature < adamM1.size(); feature++) {
            for (int neuron = 0 ;neuron < adamM1.get(feature).size(); neuron++) {
                adamM1.get(feature).set(neuron, beta1 * adamM1.get(feature).get(neuron) + (1- beta1) * gradients1.get(feature).get(neuron));
            }
        }

        for (int feature =0; feature < adamM2.size(); feature++) {
            for (int neuron = 0 ;neuron < adamM2.get(feature).size(); neuron++) {
                adamM2.get(feature).set(neuron, beta1 * adamM2.get(feature).get(neuron) + (1- beta1) * gradients2.get(feature).get(neuron));
            }
        }

        for (int feature =0; feature < adamM3.size(); feature++) {
            for (int neuron = 0 ;neuron < adamM3.get(feature).size(); neuron++) {
                adamM3.get(feature).set(neuron, beta1 * adamM3.get(feature).get(neuron) + (1- beta1) * gradients3.get(feature).get(neuron));
            }
        }

        //2. update adamV

        for (int feature =0; feature < adamV1.size(); feature++) {
            for (int neuron = 0 ;neuron < adamV1.get(feature).size(); neuron++) {
                adamV1.get(feature).set(neuron, beta2 * adamV1.get(feature).get(neuron) + (1 - beta2) * Math.pow(gradients1.get(feature).get(neuron), 2));
            }
        }

        for (int feature =0; feature < adamV2.size(); feature++) {
            for (int neuron = 0 ;neuron < adamV2.get(feature).size(); neuron++) {
                adamV2.get(feature).set(neuron, beta2 * adamV2.get(feature).get(neuron) + (1 - beta2) * Math.pow(gradients2.get(feature).get(neuron), 2));
            }
        }

        for (int feature =0; feature < adamV3.size(); feature++) {
            for (int neuron = 0 ;neuron < adamV3.get(feature).size(); neuron++) {
                adamV3.get(feature).set(neuron, beta2 * adamV3.get(feature).get(neuron) + (1 - beta2) * Math.pow(gradients3.get(feature).get(neuron), 2));
            }
        }

        //3. scale adamM

        for (int feature = 0; feature < adamM1.size(); feature++) {
            adamM1.set(feature, scalarDivide(adamM1.get(feature), (1 - Math.pow(beta1, epoch))));
        }

        for (int feature = 0; feature < adamM2.size(); feature++) {
            adamM2.set(feature, scalarDivide(adamM2.get(feature), (1 - Math.pow(beta1, epoch))));
        }

        for (int feature = 0; feature < adamM3.size(); feature++) {
            adamM3.set(feature, scalarDivide(adamM3.get(feature), (1 - Math.pow(beta1, epoch))));
        }

        //4. scale adamV

        for (int feature = 0; feature < adamV1.size(); feature++) {
            adamV1.set(feature, scalarDivide(adamV1.get(feature), (1 - Math.pow(beta2, epoch))));
        }

        for (int feature = 0; feature < adamV2.size(); feature++) {
            adamV2.set(feature, scalarDivide(adamV2.get(feature), (1 - Math.pow(beta2, epoch))));
        }

        for (int feature = 0; feature < adamV3.size(); feature++) {
            adamV3.set(feature, scalarDivide(adamV3.get(feature), (1 - Math.pow(beta2, epoch))));
        }

        //5. update the weights!

        ArrayList<ArrayList<Double>> scaledMomentum1 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> scaledMomentum2 = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> scaledMomentum3 = new ArrayList<ArrayList<Double>>();
        for (int feature = 0; feature < weights1.size(); feature++) {
            ArrayList<Double> currSM = new ArrayList<Double>();
            for (int neuron =0 ; neuron < weights1.get(feature).size(); neuron++) {
                currSM.add(adamM1.get(feature).get(neuron) / Math.sqrt(adamV1.get(feature).get(neuron) + e));
            }
            scaledMomentum1.add(currSM);
        }

        for (int feature = 0; feature < weights2.size(); feature++) {
            ArrayList<Double> currSM = new ArrayList<Double>();
            for (int neuron =0 ; neuron < weights2.get(feature).size(); neuron++) {
                currSM.add(adamM2.get(feature).get(neuron) / Math.sqrt(adamV2.get(feature).get(neuron) + e));
            }
            scaledMomentum2.add(currSM);
        }

        for (int feature = 0; feature < weights3.size(); feature++) {
            ArrayList<Double> currSM = new ArrayList<Double>();
            for (int neuron =0 ; neuron < weights3.get(feature).size(); neuron++) {
                currSM.add(adamM3.get(feature).get(neuron) / Math.sqrt(adamV3.get(feature).get(neuron) + e));
            }
            scaledMomentum3.add(currSM);
        }

        for (int feature = 0; feature < weights1.size(); feature++) {
            for (int neuron = 0; neuron < weights1.get(feature).size(); neuron++) {
                weights1.get(feature).set(neuron, weights1.get(feature).get(neuron) - learningRate * scaledMomentum1.get(feature).get(neuron));
            }
        }

        for (int feature =0 ; feature < weights2.size(); feature++) {
            for (int neuron =0; neuron < weights2.get(feature).size(); neuron++) {
                weights2.get(feature).set(neuron, weights2.get(feature).get(neuron) - learningRate * scaledMomentum2.get(feature).get(neuron));
            }
        }

        for (int feature =0; feature < weights3.size(); feature++) {
            for (int neuron = 0 ; neuron < weights3.get(feature).size(); neuron++) {
                weights3.get(feature).set(neuron, weights3.get(feature).get(neuron) - learningRate * scaledMomentum3.get(feature).get(neuron));
            }
        }

        //for the biases!

        //1 & 3. update  + scale adamBM
        for (int neurons = 0; neurons< adamBM1.size(); neurons++) {
            adamBM1.set(neurons, vectorAdd(scalarMultiply(adamBM1.get(neurons), biasBeta1), scalarMultiply(bgrads1.get(neurons), (1-biasBeta1))));
            adamBM1.set(neurons, scalarDivide(adamBM1.get(neurons), (1-Math.pow(biasBeta1, epoch))));
        }


        for (int neurons = 0; neurons< adamBM2.size(); neurons++) {
            adamBM2.set(neurons, vectorAdd(scalarMultiply(adamBM2.get(neurons), biasBeta1), scalarMultiply(bgrads2.get(neurons), (1-biasBeta1))));
            adamBM2.set(neurons, scalarDivide(adamBM2.get(neurons), (1-Math.pow(biasBeta1, epoch))));
        }

        for (int neurons = 0; neurons< adamBM3.size(); neurons++) {
            adamBM3.set(neurons, vectorAdd(scalarMultiply(adamBM3.get(neurons), biasBeta1), scalarMultiply(bgrads3.get(neurons), (1-biasBeta1))));
            adamBM3.set(neurons, scalarDivide(adamBM3.get(neurons), (1-Math.pow(biasBeta1, epoch))));
        }

        //2 & 4. update + scale adamBV

        for (int neurons = 0; neurons < adamBV1.size(); neurons++) {
            adamBV1.set(neurons, vectorAdd(scalarMultiply(adamBV1.get(neurons), biasBeta2), scalarMultiply(vectorSquared(bgrads1.get(neurons)), (1-biasBeta2))));
            adamBV1.set(neurons, scalarDivide(adamBV1.get(neurons), (1-Math.pow(biasBeta2, epoch))));
        }

        for (int neurons = 0; neurons < adamBV2.size(); neurons++) {
            adamBV2.set(neurons, vectorAdd(scalarMultiply(adamBV2.get(neurons), biasBeta2), scalarMultiply(vectorSquared(bgrads2.get(neurons)), (1-biasBeta2))));
            adamBV2.set(neurons, scalarDivide(adamBV2.get(neurons), (1-Math.pow(biasBeta2, epoch))));
        }

        for (int neurons = 0; neurons < adamBV3.size(); neurons++) {
            adamBV3.set(neurons, vectorAdd(scalarMultiply(adamBV3.get(neurons), biasBeta2), scalarMultiply(vectorSquared(bgrads3.get(neurons)), (1-biasBeta2))));
            adamBV3.set(neurons, scalarDivide(adamBV3.get(neurons), (1-Math.pow(biasBeta2, epoch))));
        }

        //5. update biases

        scaledMomentum1 = new ArrayList<ArrayList<Double>>();
        scaledMomentum2 = new ArrayList<ArrayList<Double>>();
        scaledMomentum3 = new ArrayList<ArrayList<Double>>();

        for (int neurons = 0; neurons < adamBM1.size(); neurons++) {
            scaledMomentum1.add(elementWiseDiv(adamBM1.get(neurons), vectorSqrt(scalarAdd(adamBV1.get(neurons), e))));
        }

        for (int neurons = 0; neurons < adamBM2.size(); neurons++) {
            scaledMomentum2.add(elementWiseDiv(adamBM2.get(neurons), vectorSqrt(scalarAdd(adamBV2.get(neurons), e))));
        }

        for (int neurons = 0; neurons < adamBM3.size(); neurons++) {
            scaledMomentum3.add(elementWiseDiv(adamBM3.get(neurons), vectorSqrt(scalarAdd(adamBV3.get(neurons), e))));
        }

        for (int neurons =0; neurons < scaledMomentum1.size(); neurons++) {
            biases1 = vectorSubtract(biases1, scalarMultiply(scaledMomentum1.get(neurons), learningRate));
        }

        for (int neurons =  0; neurons < scaledMomentum2.size(); neurons++) {
            biases2 = vectorSubtract(biases2, scalarMultiply(scaledMomentum2.get(neurons), learningRate));
        }

        for (int neurons = 0 ; neurons < scaledMomentum3.size(); neurons++) {
            biases3 = vectorSubtract(biases3, scalarMultiply(scaledMomentum3.get(neurons), learningRate));
        }
    }

    public static void nesterov() {
        for (int feature = 0; feature < weights1.size(); feature++) {
            for(int neuron = 0; neuron < weights1.get(feature).size(); neuron++) {
                weights1.get(feature).set(neuron, weights1.get(feature).get(neuron) - momentum * velocity1.get(feature).get(neuron));
            }
        }

        for (int feature = 0; feature < weights2.size(); feature++) {
            for(int neuron = 0; neuron < weights2.get(feature).size(); neuron++) {
                weights2.get(feature).set(neuron, weights2.get(feature).get(neuron) - momentum * velocity2.get(feature).get(neuron));
            }
        }

        for (int feature = 0; feature < weights3.size(); feature++) {
            for(int neuron = 0; neuron < weights3.get(feature).size(); neuron++) {
                weights3.get(feature).set(neuron, weights3.get(feature).get(neuron) - momentum * velocity3.get(feature).get(neuron));
            }
        }
    }

    public static void multiThreadTrain() throws Exception{
        ArrayList<ArrayList<ArrayList<Double>>> miniBatches = new ArrayList<ArrayList<ArrayList<Double>>>();
        ArrayList<ArrayList<ArrayList<Double>>> miniBatchesLabels = new ArrayList<ArrayList<ArrayList<Double>>>();
        ArrayList<ArrayList<Double>> currentData = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> currentLabels  = new ArrayList<ArrayList<Double>>();
        for (int i =0; i < x_train.size(); i++) {
            if (currentData.size() == miniBatchSize) {
                miniBatches.add(currentData);
                miniBatchesLabels.add(currentLabels);
                currentData = new ArrayList<ArrayList<Double>>();
                currentLabels = new ArrayList<ArrayList<Double>>();
            }
            else {
                currentData.add(x_train.get(i));
                currentLabels.add(y_train.get(i));
            }

            if (currentData.size() != miniBatchSize && i == x_train.size()) {
                miniBatches.add(currentData);
                miniBatchesLabels.add(currentLabels);
                currentData = new ArrayList<ArrayList<Double>>();
                currentLabels = new ArrayList<ArrayList<Double>>();
            }
        }

        System.out.println("Mini batches size : ");
        for (int i =0; i < miniBatches.size(); i++) {
            System.out.println("LENGTH : " + miniBatches.get(i).size() + " " + miniBatchesLabels.get(i).size());
        }
        Thread.sleep(6000);

        ExecutorService threadPool = Executors.newFixedThreadPool(7);
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (int i =0; i < miniBatches.size(); i++) {
                int finalI = i;
                int finalEpoch = epoch;
                threadPool.submit(() -> {
                    try {
                        weightUpdate(miniBatches.get(finalI), miniBatchesLabels.get(finalI), (double) finalEpoch);
                    } catch (Exception exception) {
                        exception.printStackTrace();
                    }
                });
            }
        }

        threadPool.shutdown();

    }

    public static void weightUpdate(ArrayList<ArrayList<Double>> miniBatch, ArrayList<ArrayList<Double>> miniBatchLabel, double epoch) throws Exception{
        HashMap<String, Object> allForward = forwardPropagation(miniBatch, weights1, weights2, weights3);
        ArrayList<ArrayList<Double>> activation3 = (ArrayList<ArrayList<Double>>) allForward.get("activation3");
        ArrayList<ArrayList<Double>> activation2 = (ArrayList<ArrayList<Double>>) allForward.get("activation2");
        ArrayList<ArrayList<Double>> activation1 = (ArrayList<ArrayList<Double>>) allForward.get("activation1");
        ArrayList<ArrayList<Double>> rawA3 = (ArrayList<ArrayList<Double>>) allForward.get("rawA3");

        HashMap<String, Object> gradients = backpropagation(miniBatchLabel, activation3, rawA3, activation2, activation1);
        ArrayList<ArrayList<Double>> gradients1 = (ArrayList<ArrayList<Double>>) gradients.get("gradients1");
        ArrayList<ArrayList<Double>> gradients2 = (ArrayList<ArrayList<Double>>) gradients.get("gradients2");
        ArrayList<ArrayList<Double>> gradients3 = (ArrayList<ArrayList<Double>>) gradients.get("gradients3");
        ArrayList<ArrayList<Double>> bgradients1 = (ArrayList<ArrayList<Double>>) gradients.get("bgradients1");
        ArrayList<ArrayList<Double>> bgradients2 = (ArrayList<ArrayList<Double>>) gradients.get("bgradients2");
        ArrayList<ArrayList<Double>> bgradients3 = (ArrayList<ArrayList<Double>>) gradients.get("bgradients3");

        HashMap<String, Object> allTestForward = forwardPropagation(x_test, weights1, weights2, weights3);
        ArrayList<ArrayList<Double>> testA3 = (ArrayList<ArrayList<Double>>) allTestForward.get("activation3");
        System.out.println("grad1 : " + gradients1);
        System.out.println("grad2 : " + gradients2);
        System.out.println("grad3 : " + gradients3);

        if (momentum == 0 && biasMomentum == 0) {
            double m = miniBatch.size();
            for (int feature = 0; feature < weights3.size(); feature++) {
                for (int neuron = 0; neuron < weights3.get(feature).size(); neuron++) {
                    weights3.get(feature).set(neuron, weights3.get(feature).get(neuron) - (learningRate/m) * gradients3.get(feature).get(neuron));
                }
            }

            for (int feature = 0; feature < weights2.size(); feature++) {
                for (int neuron = 0; neuron < weights2.get(feature).size(); neuron++) {
                    weights2.get(feature).set(neuron, weights2.get(feature).get(neuron) - (learningRate/m) * gradients2.get(feature).get(neuron));
                }
            }

            for (int feature = 0; feature < weights1.size(); feature++) {
                for (int neuron = 0; neuron < weights1.get(feature).size(); neuron++) {
                    weights1.get(feature).set(neuron, weights1.get(feature).get(neuron) - (learningRate/m) * gradients1.get(feature).get(neuron));
                }
            }

            for (int bgrad = 0; bgrad < bgradients1.size(); bgrad++) {
                biases1 = vectorSubtract(biases1, scalarMultiply(bgradients1.get(bgrad), biasLearningRate/m));
            }

            for (int bgrad = 0; bgrad < bgradients2.size(); bgrad++) {
                biases2 = vectorSubtract(biases2, scalarMultiply(bgradients2.get(bgrad), biasLearningRate/m));
            }

            for (int bgrad = 0; bgrad < bgradients3.size(); bgrad++) {
                biases3 = vectorSubtract(biases3, scalarMultiply(bgradients3.get(bgrad), biasLearningRate/m));
            }
        }



        if (biasMomentum > 0 || momentum > 0) {
            System.out.println("bgradients1.size() : " + bgradients1.size() + " " + biasVelocity1.size());
            useMomentum(gradients1, gradients2, gradients3, bgradients1, bgradients2, bgradients3);
        }

        if (adam) {
            adam(gradients1, gradients2, gradients3, bgradients1, bgradients2, bgradients3, epoch, miniBatch.size());
        }


        double trainCorrect = amountCorrect(activation3, miniBatchLabel) * 100 / miniBatchSize;
        double valCorrect = amountCorrect(testA3, y_test) * 100 / y_test.size();
        System.out.println("Correct training : " + trainCorrect);
        System.out.println("Correct validation : " + valCorrect);
        ArrayList<Double> pastCorrect = new ArrayList<Double>();
        pastCorrect.add(trainCorrect);
        pastCorrect.add(valCorrect);
        pastCorrects.add(pastCorrect);
        System.out.println("Past corrects : " + pastCorrects);

        ArrayList<Double> pastLoss = new ArrayList<Double>();
        pastLoss.add(getLosses(activation3, miniBatchLabel));
        pastLoss.add(getLosses(testA3, y_test));
        pastLosses.add(pastLoss);
        System.out.println("Past losses : " + pastLosses);
        Thread.sleep(3000);
    }

    public static void main(String[]args) throws Exception{
        getTrainData();
        getValidationData();
        if (adam) {adamInit();}
        paramInit();
        XavierInit();
        biasMomentumInit();

        System.out.println("Shape : " + weights1.size() + " " + weights1.get(0).size());
        System.out.println("Shape : " + weights2.size() + " " + weights2.get(0).size());
        System.out.println("Shape : " + weights3.size() + " " + weights3.get(0).size());
        System.out.println("Biases : " + biases1.size() + " " + biases2.size() + " " + biases3.size());

        System.out.println("VAR 1 : " + variance(ReplModelingRNN.flatten2D(weights1)));
        System.out.println("VAR 2 : " + variance(ReplModelingRNN.flatten2D(weights2)));
        System.out.println("VAR 3 : " + variance(ReplModelingRNN.flatten2D(weights3)));

        /*HashMap<String, Object> allForward = forwardPropagation(x_train, weights1, weights2, weights3);
        System.out.println("Shape : " + ((ArrayList<ArrayList<Double>>) allForward.get("activation3")).size() + " " + ((ArrayList<ArrayList<Double>>) allForward.get("activation3")).get(0).size());
        System.out.println("Shape : " + ((ArrayList<ArrayList<Double>>) allForward.get("activation2")).size() + " " + ((ArrayList<ArrayList<Double>>) allForward.get("activation2")).get(0).size());
        System.out.println("Shape : " + ((ArrayList<ArrayList<Double>>) allForward.get("activation1")).size() + " " + ((ArrayList<ArrayList<Double>>) allForward.get("activation1")).get(0).size());*/ //check activations

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            //if (epoch % 3  == 0 ) {learningRate = learningRate * 0.1; biasLearningRate = biasLearningRate * 0.1;}
            HashMap<String, Object> allForward = forwardPropagation(x_train, weights1, weights2, weights3);
            ArrayList<ArrayList<Double>> activation3 = (ArrayList<ArrayList<Double>>) allForward.get("activation3");
            ArrayList<ArrayList<Double>> activation2 = (ArrayList<ArrayList<Double>>) allForward.get("activation2");
            ArrayList<ArrayList<Double>> activation1 = (ArrayList<ArrayList<Double>>) allForward.get("activation1");
            ArrayList<ArrayList<Double>> rawA3 = (ArrayList<ArrayList<Double>>) allForward.get("rawA3");

            System.out.println("EPOCH : " + epoch);
            HashMap<String, Object> gradients = backpropagation(y_train, activation3, rawA3, activation2, activation1);
            ArrayList<ArrayList<Double>> gradients1 = (ArrayList<ArrayList<Double>>) gradients.get("gradients1");
            ArrayList<ArrayList<Double>> gradients2 = (ArrayList<ArrayList<Double>>) gradients.get("gradients2");
            ArrayList<ArrayList<Double>> gradients3 = (ArrayList<ArrayList<Double>>) gradients.get("gradients3");
            ArrayList<ArrayList<Double>> bgradients1 = (ArrayList<ArrayList<Double>>) gradients.get("bgradients1");
            ArrayList<ArrayList<Double>> bgradients2 = (ArrayList<ArrayList<Double>>) gradients.get("bgradients2");
            ArrayList<ArrayList<Double>> bgradients3 = (ArrayList<ArrayList<Double>>) gradients.get("bgradients3");

            HashMap<String, Object> allTestForward = forwardPropagation(x_test, weights1, weights2, weights3);
            ArrayList<ArrayList<Double>> testA3 = (ArrayList<ArrayList<Double>>)  allTestForward.get("activation3");
            System.out.println("grad1 : " + gradients1);
            System.out.println("grad2 : " + gradients2);
            System.out.println("grad3 : " + gradients3);

            double m = x_train.size();
            if (momentum == 0 && biasMomentum == 0) {
                for (int feature = 0; feature < weights3.size(); feature++) {
                    for (int neuron = 0; neuron < weights3.get(feature).size(); neuron++) {
                        weights3.get(feature).set(neuron, weights3.get(feature).get(neuron) - (learningRate/m) * gradients3.get(feature).get(neuron));
                    }
                }

                for (int feature = 0; feature < weights2.size(); feature++) {
                    for (int neuron = 0; neuron < weights2.get(feature).size(); neuron++) {
                        weights2.get(feature).set(neuron, weights2.get(feature).get(neuron) - (learningRate/m) * gradients2.get(feature).get(neuron));
                    }
                }

                for (int feature = 0; feature < weights1.size(); feature++) {
                    for (int neuron = 0; neuron < weights1.get(feature).size(); neuron++) {
                        weights1.get(feature).set(neuron, weights1.get(feature).get(neuron) - (learningRate/m) * gradients1.get(feature).get(neuron));
                    }
                }

                for (int bgrad = 0; bgrad < bgradients1.size(); bgrad++) {
                    biases1 = vectorSubtract(biases1, scalarMultiply(bgradients1.get(bgrad), biasLearningRate/m));
                }

                for (int bgrad = 0; bgrad< bgradients2.size(); bgrad++) {
                    biases2 = vectorSubtract(biases2, scalarMultiply(bgradients2.get(bgrad), biasLearningRate/m));
                }

                for (int bgrad = 0; bgrad < bgradients3.size(); bgrad++) {
                    biases3 = vectorSubtract(biases3, scalarMultiply(bgradients3.get(bgrad), biasLearningRate/m));
                }
            }

            if (adam) {
                adam(gradients1, gradients2, gradients3, bgradients1, bgradients2, bgradients3, epoch, m);
            }
            if (momentum > 0 || biasMomentum > 0) {
                useMomentum(gradients1, gradients2, gradients3, bgradients1, bgradients2, bgradients3);
            }


            double trainCorrect = amountCorrect(activation3, y_train) * 100/y_train.size();
            double valCorrect = amountCorrect(testA3, y_test) * 100/y_test.size();
            System.out.println("Correct training : " + trainCorrect);
            System.out.println("Correct validation : " + valCorrect);
            ArrayList<Double> pastCorrect = new ArrayList<Double>();
            pastCorrect.add(trainCorrect);
            pastCorrect.add(valCorrect);
            pastCorrects.add(pastCorrect);
            System.out.println("Past corrects : " + pastCorrects);

            ArrayList<Double> pastLoss = new ArrayList<Double>();
            pastLoss.add(getLosses(activation3, y_train));
            pastLoss.add(getLosses(testA3, y_test));
            pastLosses.add(pastLoss);
            System.out.println("Past losses : " + pastLosses);
            Thread.sleep(3000);


        }
        saveWeights();
    }

    public static void saveWeights() throws Exception {
        FileOutputStream fos = new FileOutputStream("weights1_20k_GD.tmp");
        ObjectOutputStream oos = new ObjectOutputStream(fos);
        oos.writeObject(weights1);

        fos = new FileOutputStream("weights2_20k_Adam128.tmp");
        oos = new ObjectOutputStream(fos);
        oos.writeObject(weights2);

        fos = new FileOutputStream("weights3_20k_Adam128.tmp");
        oos = new ObjectOutputStream(fos);
        oos.writeObject(weights3);

        fos = new FileOutputStream("biases1_20k_Adam128.tmp");
        oos = new ObjectOutputStream(fos);
        oos.writeObject(biases1);

        fos = new FileOutputStream("biases2_20k_Adam128.tmp");
        oos = new ObjectOutputStream(fos);
        oos.writeObject(biases2);


        fos = new FileOutputStream("biases3_20k_Adam128.tmp");
        oos = new ObjectOutputStream(fos);
        oos.writeObject(biases3);

        oos.close();
    }

    public static ArrayList<Integer> shape(ArrayList<ArrayList<Double>> x ) {
        ArrayList<Integer> full = new ArrayList<Integer>();
        full.add(x.size());
        full.add(x.get(0).size());
        return full;
    }

    public static ArrayList<Double> mostCommonLabels (ArrayList<ArrayList<Double>> a3) {
        ArrayList<Double> newPred = new ArrayList<Double>();
        for (int pred = 0; pred < a3.size(); pred++) {
            double highest_pred = 0;
            for (int output = 0; output < a3.get(pred).size(); output++) {
                if (highest_pred < a3.get(pred).get(output)) {highest_pred = a3.get(pred).get(output);}
            }
            newPred.add((double) a3.get(pred).indexOf(highest_pred));
        }
        return newPred;
    }

    public static double mostCommonLabel(ArrayList<Double> activation) {
        double highest_pred = 0;
        for (int output = 0; output < activation.size(); output++) {
            if (activation.get(output) > highest_pred) {highest_pred = activation.get(output);}
        }

        return (double) activation.indexOf(highest_pred);
    }

    public static double amountCorrect(ArrayList<ArrayList<Double>> yH, ArrayList<ArrayList<Double>> labels) {
        ArrayList<Double> newYh = mostCommonLabels(yH);
        ArrayList<Double> newLabels = mostCommonLabels(labels);
        int correct = 0;
        for (int i =0; i < newYh.size(); i++) {
            if (newLabels.get(i).equals(newYh.get(i))) {correct += 1;}
        }
        return correct;
    }

    public static ArrayList<Double> getCol(ArrayList<ArrayList<Double>> x, int col) {
        ArrayList<Double> column = new ArrayList<Double>();
        for (int i = 0; i < x.size(); i++) {
            column.add(x.get(i).get(col));
        }
        return column;
    }

    public static ArrayList<ArrayList<Double>> softmaxActivation(ArrayList<ArrayList<Double>> activ) {
        ArrayList<ArrayList<Double>> newActivation = new ArrayList<ArrayList<Double>>();
        for (int i =0; i < activ.size(); i++) {
            newActivation.add(softmax(activ.get(i)));
        }
        return newActivation;
    }

    public static ArrayList<Double> softmax(ArrayList<Double> x) {
        ArrayList<Double> probabilities = new ArrayList<Double>();
        for (int output = 0; output < x.size(); output++) {
            double num = Math.exp(x.get(output));
            double denum = 0.0;
            for (int i = 0; i < x.size(); i++) {
                denum += Math.exp(x.get(i));
            }
            probabilities.add(num/denum);
        }
        return probabilities;
    }

    public static double reverseELU(double x) {
        double output = 0;
        if (x > 0) {output = x;}
        else {
            output = (x/alpha) + 1;
        }
        return output;
    }

    public static ArrayList<ArrayList<Double>> softmaxDeriv(ArrayList<Double> output) {
        ArrayList<ArrayList<Double>> jacobian = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < output.size(); i++) {
            ArrayList<Double> currentRow = new ArrayList<Double>();
            for (int j = 0 ; j < output.size(); j++) {
                if (i == j) {
                    currentRow.add(-1 * output.get(i) * (1 - output.get(j)));
                }
                else {
                    currentRow.add(-1 * output.get(i) * output.get(j));
                }
            }
            jacobian.add(currentRow);
        }
        return jacobian;
    }

    public static ArrayList<ArrayList<Double>> transpose(ArrayList<ArrayList<Double>> x) {
        ArrayList<ArrayList<Double>> transposeX = new ArrayList<ArrayList<Double>>();
        for (int col = 0; col < x.get(0).size(); col++) {
            transposeX.add(getCol(x, col));
        }
        return transposeX;
    }

    public static ArrayList<Double> elementWiseDiv(ArrayList<Double> x, ArrayList<Double> y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i)/y.get(i));
        }
        return z;
    }

    public static ArrayList<Double> scalarMultiply(ArrayList<Double> vector, double x) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i = 0; i < vector.size(); i++) {
            z.add(vector.get(i) * x);
        }
        return z;
    }


    public static ArrayList<Double> scalarDivide(ArrayList<Double> vector, double x) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i = 0; i < vector.size(); i++) {
            z.add(vector.get(i) / x);
        }
        return z;
    }

    public static ArrayList<Double> scalarAdd(ArrayList<Double> vector, double x) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i = 0; i < vector.size(); i++) {
            z.add(vector.get(i) + x);
        }
        return z;
    }

    public static ArrayList<Double> dot(ArrayList<ArrayList<Double>> x, ArrayList<Double> y) {
        ArrayList<Double> output = new ArrayList<Double>();
        for (int i = 0; i < x.size(); i++) {
            output.add(dotSum(x.get(i), y));
        }
        return output;
    }

    public static double dotSum(ArrayList<Double> x, ArrayList<Double> y) {
        double sum = 0.0;
        for (int i = 0; i < x.size(); i++) {
            sum += x.get(i) * y.get(i);
        }
        return sum;
    }

    public static ArrayList<ArrayList<Double>> biasAdd(ArrayList<ArrayList<Double>> outputs, ArrayList<Double> biases) {
        ArrayList<ArrayList<Double>> newOutputs = new ArrayList<ArrayList<Double>>();
        for (int output = 0; output < outputs.size(); output++) {
            newOutputs.add(vectorAdd(outputs.get(output), biases));
        }
        return newOutputs;
    }

    public static ArrayList<Double> vectorAdd(ArrayList<Double> x, ArrayList<Double> y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i) + y.get(i));
        }
        return z;
    }

    public static ArrayList<Double> vectorSubtract(ArrayList<Double> x, ArrayList<Double> y) {
        ArrayList<Double> z = new ArrayList<Double>();
        for (int i =0; i < x.size(); i++) {
            z.add(x.get(i) - y.get(i));
        }
        return z;
    }

    public static double variance(ArrayList<Double> x) {
        //first get mean
        double meanSum = 0;
        for (int i = 0; i < x.size(); i++) {meanSum += x.get(i);}
        double mean = meanSum/x.size();

        double numerator = 0;
        for (int i = 0; i < x.size(); i++) {numerator += (x.get(i) - mean) * (x.get(i) - mean);}
        return numerator/x.size();
    }

    public static ArrayList<ArrayList<Double>> gradClip(ArrayList<ArrayList<Double>> grads) {
        for (int feature = 0 ; feature < grads.size(); feature++) {
            for (int neuron = 0; neuron < grads.get(feature).size(); neuron++) {
                if (grads.get(feature).get(neuron) > gradClip) {
                    grads.get(feature).set(neuron, gradClip);
                }
                if (grads.get(feature).get(neuron) < -1 * gradClip) {
                    grads.get(feature).set(neuron, -1 * gradClip);
                }
            }
        }
        return grads;
    }


}

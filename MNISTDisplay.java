package MNIST_DNN;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.*;
import java.io.*;
import java.lang.*;
import java.util.HashMap;

/**
 *
 * @author paul
 */

/*public class MNIST_DNN.MNISTDisplay {
    public int lastx = 0;
    public int lasty = 0;
    //place images into icons

    public MNIST_DNN.MNISTDisplay() {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                JFrame frame = new JFrame("Image viewer");
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                BufferedImage img = null;

                try {
                    img = ImageIO.read(this.getClass().getResource("/users/anish/downloads/IMG_2939.JPG"));
                } catch (IOException e) {
                    e.printStackTrace();
                }

                System.out.println("This is image : "  + img);
                ImageIcon imgIcon = new ImageIcon(img);
                JLabel label = new JLabel();
                label.setIcon(imgIcon);
                frame.getContentPane().add(label, BorderLayout.CENTER);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            }
        });
    }

    public static void main(String[]args) {
        new MNIST_DNN.MNISTDisplay();
    }
}

public class MNIST_DNN.MNISTDisplay extends Canvas {

    public void paint(Graphics g) {

        Toolkit t=Toolkit.getDefaultToolkit();
        Image i=t.getImage("/users/anish/downloads/IMG_1290.JPG");
        g.drawImage(i, 120,100,this);
    }
    public static void main(String[] args) {
        MNIST_DNN.MNISTDisplay m = new MNIST_DNN.MNISTDisplay();
        JFrame f=new JFrame();
        f.add(m);
        f.setSize(400,400);
        f.setVisible(true);
    }

}*/

public class MNISTDisplay extends JFrame{

    public static Toolkit t = Toolkit.getDefaultToolkit();
    public static String currImg = "dwa";
    public static MNISTDisplay frame = new MNISTDisplay();
    public static Graphics g;
    public static BufferedImage image;
    public static JLabel predictionLabel = new JLabel();
    public static JLabel accuracyLabel = new JLabel();
    public static int amountCorrect = 0;
    /*static {
        try {
            //image = ImageIO.read(new File("/users/anish/downloads/MNIST_testSetJPG/IMAGE" + (0) + ".jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }*/

    public static ArrayList<ArrayList<Double>> x_test = new ArrayList<ArrayList<Double>>();
    public static ArrayList<ArrayList<Double>> y_test = new ArrayList<ArrayList<Double>>();
    public static int numVal = 100;
    public static void getValidationData() throws Exception {
        String train_path = "/Users/anish/Documents/Java Fun/ML Java/src/MNIST_DNN/mnist_test.csv";
        String line = "";
        BufferedReader br = new BufferedReader(new FileReader(train_path));
        int z = numVal;
        while ((line = br.readLine()) != null) {
            String[] values = line.split(",");
            ArrayList<Double> currentX_train = new ArrayList<Double>();
            for (int i = 0; i < values.length; i++) {
                if (i == 0) {y_test.add(FinalMNIST.oneHot(Double.parseDouble(values[0])));}
                else {
                    currentX_train.add((Double.parseDouble(values[i]))/255.0);
                }
            }
            x_test.add(currentX_train);
            z -= 1;
            if (z == 0) {break;}
        }
    }

    public static ArrayList<Double> mcY_test = FinalMNIST.mostCommonLabels(y_test);



    //todo optmize paint() code
    public  void paint(Graphics g) {
        g.drawImage(image, 200,  200, 28 , 28, null);
        System.out.println("Done actual");
    }


    public static void main(String[]args) throws Exception{
        getValidationData();
        mcY_test = FinalMNIST.mostCommonLabels(y_test);

        //first read everything in
        FileInputStream fis = new FileInputStream("/Users/anish/Documents/Java Fun/ML Java/src/MNIST_DNN/params/weights1_10k_GDLOCAL4_7.tmp");
        ObjectInputStream ois = new ObjectInputStream(fis);
        ArrayList<ArrayList<Double>> weights1 = (ArrayList<ArrayList<Double>>) ois.readObject();

        fis = new FileInputStream("/Users/anish/Documents/Java Fun/ML Java/src/MNIST_DNN/params/weights2_10k_GDLOCAL4_7.tmp");
        ois = new ObjectInputStream(fis);
        ArrayList<ArrayList<Double>> weights2 = (ArrayList<ArrayList<Double>>) ois.readObject();

        fis = new FileInputStream("/Users/anish/Documents/Java Fun/ML Java/src/MNIST_DNN/params/weights3_10k_GDLOCAL4_7.tmp");
        ois = new ObjectInputStream(fis);
        ArrayList<ArrayList<Double>> weights3 = (ArrayList<ArrayList<Double>>) ois.readObject();

        fis = new FileInputStream("/Users/anish/Documents/Java Fun/ML Java/src/MNIST_DNN/params/biases1_10k_GDLOCAL4_7.tmp");
        ois = new ObjectInputStream(fis);
        ArrayList<Double> biases1 = (ArrayList<Double>) ois.readObject();

        fis = new FileInputStream("/Users/anish/Documents/Java Fun/ML Java/src/MNIST_DNN/params/biases2_10k_GDLOCAL4_7.tmp");
        ois = new ObjectInputStream(fis);
        ArrayList<Double> biases2 = (ArrayList<Double>) ois.readObject();

        fis = new FileInputStream("/Users/anish/Documents/Java Fun/ML Java/src/MNIST_DNN/params/biases3_10k_GDLOCAL4_7.tmp");
        ois = new ObjectInputStream(fis);
        ArrayList<Double> biases3 = (ArrayList<Double>) ois.readObject();

        ois.close();

        System.out.println("W1 : " + weights1);
        System.out.println("W2 : " + weights2);
        System.out.println("W3 : " + weights3);
        System.out.println("B1 : " + biases1);
        System.out.println("B1 : " + biases2);
        System.out.println("B1 : " + biases3);

        //next get the preds in mostCommonLabels format
        HashMap<String, Object> allForward = FinalMNIST.forwardPropagation(x_test, weights1, weights2, weights3, biases1, biases2, biases3);
        ArrayList<Double> outputs = FinalMNIST.mostCommonLabels((ArrayList<ArrayList<Double>>) allForward.get("activation3"));


        JLabel title = new JLabel("MNIST Deep Neural Network");
        title.setBounds(150, 50, 200, 50);
        frame.add(title);

        predictionLabel = new JLabel("Prediction : ___ ");
        predictionLabel.setBounds(200, 300, 300, 50);
        frame.add(predictionLabel);

        accuracyLabel = new JLabel("Accuracy : ");
        accuracyLabel.setBounds(200, 360, 300, 50);
        frame.add(accuracyLabel);

        ImageIcon icon;
        JLabel imgLabel = new JLabel("Image goes here");
        imgLabel.setBounds(200, 200, 100, 100);
        frame.add(imgLabel);

        frame.pack();
        frame.setSize(500, 500);
        frame.setVisible(true);

        System.out.println("Done with finding the mostCommonLabels");
        System.out.println("outputs : " + outputs);
        System.out.println("Y_test : " + mcY_test);

        for (int output = 0; output < outputs.size(); output++) {
            //frame.getContentPane().removeAll();

            try {
                frame.getContentPane().remove(imgLabel);
            }
            catch(Exception e) {}
            //image = ImageIO.read(new File("/users/anish/downloads/MNIST_testSetJPG/IMAGE" + (output  ) + ".jpg"));
            //newImage = image.getScaledInstance(100, 100,  java.awt.Image.SCALE_SMOOTH);

            icon = new ImageIcon("/users/anish/Documents/Java Fun/ML Java/src/MNIST_DNN/MNIST_JPGs/" + (output) + ".jpg"); 
            Image img = icon.getImage();
            img = img.getScaledInstance(100, 100,  java.awt.Image.SCALE_SMOOTH);
            icon = new ImageIcon(img);
            imgLabel = new JLabel(icon);
            frame.getSize();
            imgLabel.setBounds(200, 200, (int) 0.25 * frame.getWidth(), (int) 0.25 * frame.getHeight());
            frame.add(imgLabel);
        
            predictionLabel.setText("prediction : " + outputs.get(output));
            if ((double) outputs.get(output) == (double) mcY_test.get(output)) {
                predictionLabel.setForeground(Color.green);
                amountCorrect += 1;
            }
            else {
                predictionLabel.setForeground(Color.red);
                System.out.println("FALSE : " +  outputs.get(output) + " " + mcY_test.get(output));
            }

            accuracyLabel.setText("accuracy : " + (amountCorrect * 100/(output + 1)));

            Thread.sleep(1000);
        }

    }

    public static void updatePredictionLabel(double x) {
        predictionLabel.setText("Prediction : " + x);
    }

    public static void readFile(String path) throws IOException {
        image =  ImageIO.read(new File(path));
    }

    public static double vectorMean(ArrayList<Double> x) {
        double sum = 0.0;
        for (int i =0; i < x.size(); i++) {sum += x.get(i);}
        return sum/x.size();
    }
}





package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import util.Log;

public class ConvertMushroom {
    public static void main(String[] args) {
        try {
            //create a buffered reader given the filename (which requires creating a File and FileReader object beforehand)
            BufferedReader bufferedReader = new BufferedReader(new FileReader(new File("./datasets/agaricus-lepiota.data")));
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("./datasets/agaricus-lepiota.txt")));

            String readLine = "";
            //read the file line by line
            while ((readLine = bufferedReader.readLine()) != null) {
                Log.info(readLine); //print out each line of the file if the -DLOG_LEVEL=DEBUG system property is set on the command line

                if (readLine.length() == 0 || readLine.charAt(0) == '#') {
                    //empty lines are skipped, as well as lines beginning 
                    //with the '#' character, these are comments and also skipped
                    continue;
                }

                String[] values = readLine.split(",");
                String sampleClass = values[0]; //the class of the dataset is the first column

                //put everything in a stringbuffer before writing to the file
                StringBuffer sb = new StringBuffer();

                //This dataset has three classes: 'Iris-setosa', 'Iris-versicolor', and 'Iris-virginica'
                if (sampleClass.equals("e")) {
                    sb.append("0"); //this will be the third class
                } else if (sampleClass.equals("p")) {
                    sb.append("1"); //this will be the second class
                } else {
                    System.err.println("ERROR: unknown class in mushroom.data file: '" + sampleClass + "'");
                    System.err.println("This should not happen.");
                    System.exit(1);
                }
                sb.append(":");

                //the other values are the different input values for the neural network
                for (int i = 1; i < values.length; i++) {
                    if (i > 1) sb.append(",");
                    sb.append(makeOneHot(i, values[i]));
                }
                sb.append("\n");

                Log.info(sb.toString());
                bufferedWriter.write(sb.toString());
            }
            bufferedWriter.close();
            bufferedReader.close();

        } catch (IOException e) {
            Log.fatal("ERROR converting mushroom data file");
            e.printStackTrace();
            System.exit(1);
        }
    }
    
    private static String makeOneHot(int position, String val){
        switch(position){
            case 1:
                switch(val){
                    case "b":
                        return "1,0,0,0,0,0";
                    case "c":
                        return "0,1,0,0,0,0";
                    case "x":
                        return "0,0,1,0,0,0";
                    case "f":
                        return "0,0,0,1,0,0";
                    case "k":
                        return "0,0,0,0,1,0";
                    case "s":
                        return "0,0,0,0,0,1";
                    default:
                        System.err.println("Error, unknown shape: " + val);
                        System.exit(1);
                }
            case 2:
                switch(val){
                    case "f":
                        return "1,0,0,0";
                    case "g":
                        return "0,1,0,0";
                    case "y":
                        return "0,0,1,0";
                    case "s":
                        return "0,0,0,1";
                    default:
                        System.err.println("Error, unknown surface: " + val);
                        System.exit(1);
                }
            case 3:
                switch(val){
                    case "n":
                        return "1,0,0,0,0,0,0,0,0,0";
                    case "b":
                        return "0,1,0,0,0,0,0,0,0,0";
                    case "c":
                        return "0,0,1,0,0,0,0,0,0,0";
                    case "g":
                        return "0,0,0,1,0,0,0,0,0,0";
                    case "r":
                        return "0,0,0,0,1,0,0,0,0,0";
                    case "p":
                        return "0,0,0,0,0,1,0,0,0,0";
                    case "u":
                        return "0,0,0,0,0,0,1,0,0,0";
                    case "e":
                        return "0,0,0,0,0,0,0,1,0,0";
                    case "w":
                        return "0,0,0,0,0,0,0,0,1,0";
                    case "y":
                        return "0,0,0,0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, unknown color: " + val);
                        System.exit(1);
                }
            case 4:
                switch(val){
                    case "t":
                        return "1,0";
                    case "f":
                        return "0,1";
                    default:
                        System.err.println("Error, unknown bruises value: " + val);
                        System.exit(1);
                }
            case 5:
                switch(val){
                    case "a":
                        return "1,0,0,0,0,0,0,0,0";
                    case "l":
                        return "0,1,0,0,0,0,0,0,0";
                    case "c":
                        return "0,0,1,0,0,0,0,0,0";
                    case "y":
                        return "0,0,0,1,0,0,0,0,0";
                    case "f":
                        return "0,0,0,0,1,0,0,0,0";
                    case "m":
                        return "0,0,0,0,0,1,0,0,0";
                    case "n":
                        return "0,0,0,0,0,0,1,0,0";
                    case "p":
                        return "0,0,0,0,0,0,0,1,0";
                    case "s":
                        return "0,0,0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, unknown odor: " + val);
                        System.exit(1);
                }
            case 6:
                switch(val){
                    case "a":
                        return "1,0,0,0";
                    case "d":
                        return "0,1,0,0";
                    case "f":
                        return "0,0,1,0";
                    case "n":
                        return "0,0,0,1";
                    default:
                        System.err.println("Error, unknown gill attachment: " + val);
                        System.exit(1);
                }
            case 7:
                switch(val){
                    case "c":
                        return "1,0,0";
                    case "w":
                        return "0,1,0";
                    case "d":
                        return "0,0,1";
                    default:
                        System.err.println("Error, unknown gill spacing: " + val);
                        System.exit(1);
                }
            case 8:
                switch(val){
                    case "b":
                        return "1,0";
                    case "n":
                        return "0,1";
                    default:
                        System.err.println("Error, unknown gill size: " + val);
                        System.exit(1);
                }
            case 9:
                switch(val){
                    case "k":
                        return "1,0,0,0,0,0,0,0,0,0,0,0";
                    case "n":
                        return "0,1,0,0,0,0,0,0,0,0,0,0";
                    case "b":
                        return "0,0,1,0,0,0,0,0,0,0,0,0";
                    case "h":
                        return "0,0,0,1,0,0,0,0,0,0,0,0";
                    case "g":
                        return "0,0,0,0,1,0,0,0,0,0,0,0";
                    case "r":
                        return "0,0,0,0,0,1,0,0,0,0,0,0";
                    case "o":
                        return "0,0,0,0,0,0,1,0,0,0,0,0";
                    case "p":
                        return "0,0,0,0,0,0,0,1,0,0,0,0";
                    case "u":
                        return "0,0,0,0,0,0,0,0,1,0,0,0";
                    case "e":
                        return "0,0,0,0,0,0,0,0,0,1,0,0";
                    case "w":
                        return "0,0,0,0,0,0,0,0,0,0,1,0";
                    case "y":
                        return "0,0,0,0,0,0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, unknown gill color: " + val);
                        System.exit(1);
                }
            case 10:
                switch(val){
                    case "e":
                        return "1,0";
                    case "t":
                        return "0,1";
                    default:
                        System.err.println("Error, unknown stalk shape: " + val);
                        System.exit(1);
                }
            case 11:
                switch(val){
                    case "b":
                        return "1,0,0,0,0,0,0";
                    case "c":
                        return "0,1,0,0,0,0,0";
                    case "u":
                        return "0,0,1,0,0,0,0";
                    case "e":
                        return "0,0,0,1,0,0,0";
                    case "z":
                        return "0,0,0,0,1,0,0";
                    case "r":
                        return "0,0,0,0,0,1,0";
                    case "?":
                        return "0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, unknown stalk root: " + val);
                        System.exit(1);
                }
            case 12:
                switch(val){
                    case "f":
                        return "1,0,0,0";
                    case "y":
                        return "0,1,0,0";
                    case "k":
                        return "0,0,1,0";
                    case "s":
                        return "0,0,0,1";
                    default:
                        System.err.println("Error, unknown stalk surface above ring: " + val);
                        System.exit(1);
                }
            case 13:
                switch(val){
                    case "f":
                        return "1,0,0,0";
                    case "y":
                        return "0,1,0,0";
                    case "k":
                        return "0,0,1,0";
                    case "s":
                        return "0,0,0,1";
                    default:
                        System.err.println("Error, unknown stalk surface below ring: " + val);
                        System.exit(1);
                }
            case 14:
                switch(val){
                    case "n":
                        return "1,0,0,0,0,0,0,0,0";
                    case "b":
                        return "0,1,0,0,0,0,0,0,0";
                    case "c":
                        return "0,0,1,0,0,0,0,0,0";
                    case "g":
                        return "0,0,0,1,0,0,0,0,0";
                    case "o":
                        return "0,0,0,0,1,0,0,0,0";
                    case "p":
                        return "0,0,0,0,0,1,0,0,0";
                    case "e":
                        return "0,0,0,0,0,0,1,0,0";
                    case "w":
                        return "0,0,0,0,0,0,0,1,0";
                    case "y":
                        return "0,0,0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, unknown stalk color above ring: " + val);
                        System.exit(1);
                }
            case 15:
                switch(val){
                    case "n":
                        return "1,0,0,0,0,0,0,0,0";
                    case "b":
                        return "0,1,0,0,0,0,0,0,0";
                    case "c":
                        return "0,0,1,0,0,0,0,0,0";
                    case "g":
                        return "0,0,0,1,0,0,0,0,0";
                    case "o":
                        return "0,0,0,0,1,0,0,0,0";
                    case "p":
                        return "0,0,0,0,0,1,0,0,0";
                    case "e":
                        return "0,0,0,0,0,0,1,0,0";
                    case "w":
                        return "0,0,0,0,0,0,0,1,0";
                    case "y":
                        return "0,0,0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, unknown stalk color below ring: " + val);
                        System.exit(1);
                }
            case 16:
                switch(val){
                    case "p":
                        return "1,0";
                    case "u":
                        return "0,1";
                    default:
                        System.err.println("Error, unknown veil type: " + val);
                        System.exit(1);
                }
            case 17:
                switch(val){
                    case "n":
                        return "1,0,0,0";
                    case "o":
                        return "0,1,0,0";
                    case "w":
                        return "0,0,1,0";
                    case "y":
                        return "0,0,0,1";
                    default:
                        System.err.println("Error, unknown veil color: " + val);
                        System.exit(1);
                }
            case 18:
                switch(val){
                    case "n":
                        return "1,0,0";
                    case "o":
                        return "0,1,0";
                    case "t":
                        return "0,0,1";
                    default:
                        System.err.println("Error, unknown ring number: " + val);
                        System.exit(1);
                }
            case 19:
                switch(val){
                    case "c":
                        return "1,0,0,0,0,0,0,0";
                    case "e":
                        return "0,1,0,0,0,0,0,0";
                    case "f":
                        return "0,0,1,0,0,0,0,0";
                    case "l":
                        return "0,0,0,1,0,0,0,0";
                    case "n":
                        return "0,0,0,0,1,0,0,0";
                    case "p":
                        return "0,0,0,0,0,1,0,0";
                    case "s":
                        return "0,0,0,0,0,0,1,0";
                    case "z":
                        return "0,0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, ring type: " + val);
                        System.exit(1);
                }
            case 20:
                switch(val){
                    case "k":
                        return "1,0,0,0,0,0,0,0,0";
                    case "n":
                        return "0,1,0,0,0,0,0,0,0";
                    case "b":
                        return "0,0,1,0,0,0,0,0,0";
                    case "h":
                        return "0,0,0,1,0,0,0,0,0";
                    case "r":
                        return "0,0,0,0,1,0,0,0,0";
                    case "o":
                        return "0,0,0,0,0,1,0,0,0";
                    case "u":
                        return "0,0,0,0,0,0,1,0,0";
                    case "w":
                        return "0,0,0,0,0,0,0,1,0";
                    case "y":
                        return "0,0,0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, unknown spore print color: " + val);
                        System.exit(1);
                }
            case 21:
                switch(val){
                    case "a":
                        return "1,0,0,0,0,0";
                    case "c":
                        return "0,1,0,0,0,0";
                    case "n":
                        return "0,0,1,0,0,0";
                    case "s":
                        return "0,0,0,1,0,0";
                    case "v":
                        return "0,0,0,0,1,0";
                    case "y":
                        return "0,0,0,0,0,1";
                    default:
                        System.err.println("Error, population: " + val);
                        System.exit(1);
                }
            case 22:
                switch(val){
                    case "g":
                        return "1,0,0,0,0,0,0";
                    case "l":
                        return "0,1,0,0,0,0,0";
                    case "m":
                        return "0,0,1,0,0,0,0";
                    case "p":
                        return "0,0,0,1,0,0,0";
                    case "u":
                        return "0,0,0,0,1,0,0";
                    case "w":
                        return "0,0,0,0,0,1,0";
                    case "d":
                        return "0,0,0,0,0,0,1";
                    default:
                        System.err.println("Error, population: " + val);
                        System.exit(1);
                }
            default:
                System.err.println("Error, there should only be 22 inputs in the Mushroom dataset");
                System.exit(1);
                return null;
        }
    }
}

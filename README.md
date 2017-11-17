# Sentiment_Analysis

## Project Overview

In the digital world, we have become accustomed to sharing our sentiments across mediums such as Facebook, Instagram, and Twitter, to anyone that is listening. It’s a mass expressing of the pain, hope, joy, and love that can both inspire change or commence destruction.  As humans we crave the information, we are pack animals by nature and both consciously and unconsciously alter our decision based on the mood of the herd. As a result, social media has become a very powerful marketing tool, businesses bombard the platforms with advertisements with the hopes of their own finical gain. The goal of this project is to remove this muddying of the waters from a thread of Twitter messages. 
### Problem Statement, Evaluation Metrics, and Benchmarks
#### Problem Statement
Using a combination of supervised and unsupervised algorithms we hope to analyze the sentiment of any given Twitter message on a scale of Genuine Expression to Finically Motivated. The final result will be a binary classification of any message as Genuine or Motivated. 
#### Evaluation Metrics
To evaluate our solution will we use the score method from our supervised learning algorithm on a validation set is taken from a conglomerate of unique messages. 
#### Benchmarks
Because our solution is binary (Genuine or Motivated) our benchmark will be random guessing. If labels were assigned at random the program should score about 50%. 
Analysis

## Execution

To run this program open the Main.py and execute. 
After the program has run the labeled output will be saved to df_Machine_Labeled.pkl in the working directory

To change the JSON file or number of entries from the JSON file that are loaded into the DataFrame. Open the json_management.py and toggle the constants at the top of the script.

Note: This program retrains with every execution. If a JSON file does not contain the same entries at the labeled.py file the results will not be as expected.

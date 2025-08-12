# Counterfactual Samples Constructing 
Build a word contribution calculator and a counterfactual sample generator based on these two scripts, and apply them to your own model training.



### Steps

1. Mount the word contribution calculation method to the model to be trained.

2. Perform the first round of training normally.
3. Calculate the contribution of word vectors to the results in the sentences from the previous training round.
4. Generate corresponding counterfactual samples using the counterfactual generator based on the word contributions from the previous training round.
5. Use the counterfactual samples for the next round of contrastive learning training.
6. Repeat steps 3, 4, and 5.

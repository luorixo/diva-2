1. Generate many synthetic datasets -> save these locally in a folder somewhere
2. Split data into test/train sets -> save those seperately
3. Poison/add noise to the training sets -> train model on clean data, save its accuracy. Train model on poisoned data, save its accuracy. (can save clean + posisoned model accuracy together if wanted). Generated test datasets should be able to be posioned by any number/combination of poisoners.
4. Compute the complexity measures of the poisoned datasets -> save that
5. Create a dataframe where complexity measures (as a 1D array) map to its clean accuracy score: c_measure -> accuracy_clean. This should be a large dataframe where each entry is (c_measure -> accuracy_clean). Save this as a the meta database
6. Train a meta-learner on the meta-database. Whatever classifier is alright (just use the one DIVA uses).
7. We have a meta-learner, now test out DIVA.
8. Generate more synthetic datasets
9. Poison them
10. Compute their C-measures
11. Feed C-measures into the meta-learner which will spit out an estimated accuracy that our classifier should reach
12. Actually train a classifier on the poisoned data and save its accuracy
13. Compare its accuracy to the meta-learner's estimated accuracy, if large it is poisoned.

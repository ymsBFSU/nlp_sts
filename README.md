# nlp_sts
This is a project for building a Semantic Textual Similarity (STS) model from the Task 6 of SemEval 2012

## Conclusions of the current work

The final correlation between the golden standard of the test sentences and our prediction model is __0.752__. Overall, this shows that our model can find some textual similarities between sentences and give results quite close to the real ones.

We noticed through the pre-work that some simple features were giving good results like lemmas even when we were not adding more information (features). Synsets on the other hand were not giving good results alone, that's why we used them through our own optimization of lesk algorithm for disambiguating words. Furthermore, specific sizes of ngrams were proved to be quite informative as features of our model. Finally, the use of Tf/Idf metric over bag of words gave some unexpectedly good results because of similarities in the train and test set topics. We decided directly to keep this approach as another model because of the size of its feature space. Also, we didn't want to put in the same feature matrix so different kind of metrics.

At the end, we ended up using the following transformations on the data (sentences) before applying a similarity metric:
* lemmas (as a base transformation for the rest)
* n-grams
* word substitutions using word disambiguation with our optimization of lesk.
* bag of words

At this point, we need to specify that in order to get a better representation of a sentence we substituted words with the (first) lemma of the best matched hypernym synset. We thought that the hypernym between 2 words of different senteces can be the same more frequently than the synset of the word we wanted to disabiguate. Also, in order to avoid errors we return the same word in case there is no synset for a word.

The similarity metrics we used were:
* Dice coefficient
* Jaccard distance
* Tf/Idf (it's more like a statistic

Dice coefficient values more the intersection of the sentences and uses bi-grams. It proved to give consistently better results, whenever we managed to use it, in comparison with jaccard distance.

Finally, we constructed 2 different models:
* Model1, which includes the following features:
    * Dice coefficient over lemmas
    * Dice coefficient over Lesk optimization
    * Jaccard distance over lemmas
    * Jaccard distance over lemmas without stopwords
    * Jaccard distance over lemmas of n-grams:
        * 2 words
        * 4 words
        * 3 words without stopwords

* Model2, which is the Tf/Idf statistic over bag of words.

Our problem is a regression problem. We tested different regressors and at the end we ended up using AdaBoostRegressor for Model1 and GradientBoostingRegressor for Model2.

In order to combine the models we applied an ensemble technique called stacking using K-folds, K=10, splitting our training set into 9 folds for training and 1 for prediction. In every fold we get a prediction over the prediction set; then we construct a vector with all the predictions which represents the testing error over the training folds. After doing this for both models, we train with the whole training dataset for getting the predictions/output of each model over the test set.

Then, after constructing a vector from the K-fold of each model, we use them as features for our ensemble model which will combine the results of the other 2 models we described. Like this, it would be like training over the (simulated) testing error. That technique boosted our model and made it much stronger since it combines the information from both and makes our final model learn some inaccuracies of the other 2 base models. For that model we used Extreme Gradient Boosting Regressor, which is a very famous algorithm used in most of AI challenges.

In general, we noticed that simple features we had tried during the course were giving better results than more complex ones; that's why at the end, our models don't use any kind of special features. Also, we took advantage of the fact that the source of the training and some of the test (3 out of 5) sets is the same and we used bag of words with Tf/Idf. Finally, the addition of similar transformations using the same similarity metric gave a rise to our final results.

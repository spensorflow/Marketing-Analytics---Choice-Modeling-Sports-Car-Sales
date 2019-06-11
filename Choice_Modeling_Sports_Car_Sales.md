Choice Modeling Sports Cars
================
Vic Spencer
June 11, 2019

**Modeling the choice share of sportscar purchases**
----------------------------------------------------

![Vroom Vroom!](Images/sportscar.jpg)

This project explores a dataset of sportscar purchase choices comprising 200 individuals. Using the techniques of multinomial logistic regression, as well as hierarchical multinomial logistic regression, I will develop models to predict the share of purchase decisions that consumers may make when faced with a set of sportscars, given different features. This provides a framework that could hypothetically help a sportscar manufacturer determine the best features to add to their next car in order to optimize its market share.

**Load libraries**
------------------

``` r
library(mlogit)
library(tidyverse)
library(corrplot)
library(MASS)
```

**Load and preview the dataset**
--------------------------------

The sportscar dataset has already been pivotted into a long, choice model format, displaying each affirmative choice, as well as the other alternatives they faced during a given purchase decision.

``` r
sportscar <- read.csv('Data/sportscar_choice_long.csv')

head(sportscar)
```

    ##   resp_id ques alt segment seat  trans convert price choice
    ## 1       1    1   1   basic    2 manual     yes    35      0
    ## 2       1    1   2   basic    5   auto      no    40      0
    ## 3       1    1   3   basic    5   auto      no    30      1
    ## 4       1    2   1   basic    5 manual      no    35      0
    ## 5       1    2   2   basic    2 manual      no    30      1
    ## 6       1    2   3   basic    4   auto      no    35      0

``` r
str(sportscar)
```

    ## 'data.frame':    6000 obs. of  9 variables:
    ##  $ resp_id: int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ ques   : int  1 1 1 2 2 2 3 3 3 4 ...
    ##  $ alt    : int  1 2 3 1 2 3 1 2 3 1 ...
    ##  $ segment: Factor w/ 3 levels "basic","fun",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ seat   : int  2 5 5 5 2 4 5 4 4 2 ...
    ##  $ trans  : Factor w/ 2 levels "auto","manual": 2 1 1 2 2 1 1 1 2 2 ...
    ##  $ convert: Factor w/ 2 levels "no","yes": 2 1 1 1 1 1 2 2 1 2 ...
    ##  $ price  : int  35 40 30 35 30 35 35 30 40 30 ...
    ##  $ choice : int  0 0 1 0 1 0 1 0 0 0 ...

The fields in this dataset are as follows:

<table style="width:144%;">
<colgroup>
<col width="18%" />
<col width="126%" />
</colgroup>
<thead>
<tr class="header">
<th align="left"><strong>Field</strong></th>
<th align="left"><strong>Description</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">resp_id</td>
<td align="left">The identifier of each individual in the dataset</td>
</tr>
<tr class="even">
<td align="left">ques</td>
<td align="left">The identifier of each specific purchase scenario</td>
</tr>
<tr class="odd">
<td align="left">alt</td>
<td align="left">The identifier of each alternative choice within a question</td>
</tr>
<tr class="even">
<td align="left">segment</td>
<td align="left">The commercial segment of a sportscar model ('basic', 'fun', 'racer')</td>
</tr>
<tr class="odd">
<td align="left">seat</td>
<td align="left">The number of seats in the vehicle (2, 4, 5)</td>
</tr>
<tr class="even">
<td align="left">trans</td>
<td align="left">The transmission type of the vehicle ('auto','manual')</td>
</tr>
<tr class="odd">
<td align="left">convert</td>
<td align="left">Whether or not the vehicle has a convertible top</td>
</tr>
<tr class="even">
<td align="left">price</td>
<td align="left">The sportscar price (in thousands/$)</td>
</tr>
<tr class="odd">
<td align="left">choice</td>
<td align="left">Dummy indicator of the decision made. (1 = car chosen, 0 = alternative cars chosen from)</td>
</tr>
</tbody>
</table>

**Data cleaning**
-----------------

``` r
## Convert seat variable to factor
sportscar$seat <- as.factor(sportscar$seat)

## Convert data to mlogit.data
sportscar.ml <- mlogit.data(sportscar, shape = 'long', choice = 'choice', alt.var = 'alt', varying = 4:8)
```

First, we need to code seat as a factor variable, rather than an integer as it was read in. Then, we transform the dataframe into an format that is conducive to being read by the multinomial logit model. We specify that the data is in long format, with multiple alternatives per choice displayed in various rows. Then, we identify that the values for the choice we're modeling are found in the 'choice' column, and that the variable to differentiate between alternatives in a given respondent decision is the 'alt' column.

**Exploratory analysis**
------------------------

### *Perform crosstabulations of specific features and whether or not the vehicle was chosen*

``` r
##### Transmission type
xtabs(choice ~ trans, data=sportscar.ml)
```

    ## trans
    ##   auto manual 
    ##   1328    672

The sportscars that were chosen were much more likely to have an automatic, rather than a manual transmission.

``` r
##### Number of seats
xtabs(choice ~ seat, data=sportscar.ml)
```

    ## seat
    ##   2   4   5 
    ## 608 616 776

5-seaters were chosen at a higher rate than 2- and 4-seaters, each of which were chosen at relatively even rates.

``` r
##### Convertible top
xtabs(choice ~ convert, data=sportscar.ml)
```

    ## convert
    ##   no  yes 
    ##  941 1059

Sportscars with convertible tops were more likely to be chosen than standard tops.

``` r
##### Price
xtabs(choice ~ price, data=sportscar.ml)
```

    ## price
    ##   30   35   40 
    ## 1010  666  324

Perhaps unsurprisingly, $30k vehicles were more prevalent choices than $3k vehicles, which were in turn more prevalent than $40k vehicles.

``` r
##### Segment
xtabs(choice ~ segment, data=sportscar.ml)
```

    ## segment
    ## basic   fun racer 
    ##  1280   510   210

Basic models were the predominant choice, compared to fun and racer models, with fun models chosen much more than racer models.

**Choice Modeling with Multinomial Logistic Regression**
--------------------------------------------------------

``` r
## Fit multinomial logistic regression model
sportscar_model4 <- mlogit(choice ~ 0 + seat + trans + convert + price + price:segment , data = sportscar.ml)

## Generate model summary
summary(sportscar_model4)
```

    ## 
    ## Call:
    ## mlogit(formula = choice ~ 0 + seat + trans + convert + price + 
    ##     price:segment, data = sportscar.ml, method = "nr")
    ## 
    ## Frequencies of alternatives:
    ##     1     2     3 
    ## 0.328 0.327 0.345 
    ## 
    ## nr method
    ## 5 iterations, 0h:0m:0s 
    ## g'(-H)^-1g = 0.000178 
    ## successive function values within tolerance limits 
    ## 
    ## Coefficients :
    ##                     Estimate Std. Error  z-value  Pr(>|z|)    
    ## seat4              -0.016206   0.076170  -0.2128 0.8315107    
    ## seat5               0.426851   0.075682   5.6401 1.700e-08 ***
    ## transmanual        -1.228724   0.066893 -18.3686 < 2.2e-16 ***
    ## convertyes          0.200792   0.062343   3.2207 0.0012786 ** 
    ## price              -0.228245   0.011483 -19.8771 < 2.2e-16 ***
    ## price:segmentfun    0.094360   0.019215   4.9108 9.069e-07 ***
    ## price:segmentracer  0.095829   0.025887   3.7018 0.0002141 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Log-Likelihood: -1694.9

### **Model results**

-   Seat count:
    -   Using a 2-seater as the baseline seat configuration, we can see that respondents were statistically more likely to choose 5-seater, with a confidence level of 99.9%.
-   Transmission type:
    -   Respondents were much more likely to choose automatic when asked to choose between cars with automatic and manual transmissions.
-   Convertible tops:
    -   Convertible-top car models were statistically more popular than those with standard roofs.
-   Price:
    -   Chosen cars were statistically cheaper than the alternatives
-   Price interacted with Segment
    -   When controlling for price, both the fun and racer segments were statistically more likely to be chosen at higher price points than their basic counterpart

### **Calculate Willingness-To-Pay (WTP) for a given feature**

#### *Let's explore what kind of premium (or discount) would be typically required to get a customer to choose a given feature. Note that the numbers are expressed in thousands of dollars*

``` r
WTP <- coef(sportscar_model4)/- coef(sportscar_model4)[5]
WTP
```

    ##              seat4              seat5        transmanual 
    ##        -0.07100377         1.87014240        -5.38335124 
    ##         convertyes              price   price:segmentfun 
    ##         0.87971941        -1.00000000         0.41341735 
    ## price:segmentracer 
    ##         0.41984953

-   Seats
    -   Selling a 4-seater over a 2-seater may require a modest $70 discount.
    -   On the other hand, a dealer could raise the price of a 5-seater by $1,870 and potentially sell over a 2-seat sports car.
-   Transmissions
    -   Selling a manual transmission may require a pretty steep discount of $5,3834 if attempting to sell over an automatic.
-   Convertibles
    -   A dealer could get away with selling a convertible for an additional $879 more than a sportcar with a standard roof.
-   Segment
    -   Both the fun and the racer models could potentially be priced at an additional $400+ when competing with the basic sportscar segment.

### **Create a function to clean and make predictions based on the model**

``` r
## Predict choice share based on model
predict_mnl <- function(model, products) {
  data.model <- model.matrix(update(model$formula, 0 ~ .), 
                             data = products)[,-1]
  utility <- data.model%*%model$coef
  share <- exp(utility)/sum(exp(utility))
  cbind(share, products)
}
```

### **Create scenario data to predict choice shares**

Hypothetical scenario: Let's say that a client wants to enter the 2-seat racer market (Car 1), knowing that there are 2 other big competitors in that space (Cars 2 and 3). In this scenario, Car 2 has a manual transmission with a convertible top, while Car 3 is an automatic with a standard roof, and both vehicles are priced at $30k. What choice would consumers make if our client developed a 2-seat racer with an automatic transmission and a convertible top, also priced at $30k?

``` r
## Create hypothetical data for choice share prediction
car <- c(1,2,3)
price <- c(30, 30, 30)
seat <- factor(c(2, 2, 2), levels=c(2,4,5))
trans <- factor(c("auto", "manual","auto"), levels=c("auto", "manual"))
convert <- factor(c("yes", "yes", "no"), levels=c("no", "yes"))
segment <- factor(c("racer", "racer", "racer"), levels=c("basic", "fun", "racer"))
prod <- data.frame(car,seat, trans, convert, price, segment)
```

### **Predict choice shares of 2-seat racer sportscars**

``` r
## Predict choice shares of hypothetical 3-option sports car selection
shares <- predict_mnl(sportscar_model4, prod)
shares
```

    ##       share car seat  trans convert price segment
    ## 1 0.4737655   1    2   auto     yes    30   racer
    ## 2 0.1386550   2    2 manual     yes    30   racer
    ## 3 0.3875795   3    2   auto      no    30   racer

Nice! When presented with the choice between our client's vehicle and the two competing vehicles, our client could expect a plurality of market, with a choice share of 47%.

**Plot the choice shares per car**
----------------------------------

``` r
ggplot(shares, aes(x = car, y = share, fill = car))+
  geom_bar(stat = 'identity')+
  ylab('Predicted Market Share')+
  xlab('Proposed Car Models')+
  ggtitle('Choice Share of Car Models')
```

![](Choice_Modeling_Sports_Car_Sales_files/figure-markdown_github/unnamed-chunk-12-1.png)

### **Choice Modeling with Hiercharchical Multinomial Logistic Regression**

Let's improve our model by moving to hierarchical modeling, which will allow us to take into account differences in preferences between various individuals.

#### **Prepare data for hierachical model**

``` r
# Create new version of the data to prepare to re-code the non-binary factor variables
sportscar.ml2 <- sportscar.ml <- mlogit.data(sportscar, shape = 'long', choice = 'choice', alt.var = 'alt', varying = 4:8, id.var = 'resp_id')

# Set the contrasts for non-binary factor variables to code against the effects of the baseline level
contrasts(sportscar.ml2$segment) <- contr.sum(levels(sportscar.ml2$segment))
dimnames(contrasts(sportscar.ml2$segment))[[2]] <- levels(sportscar.ml$segment)[1:2]

contrasts(sportscar.ml2$seat) <- contr.sum(levels(sportscar.ml2$seat))
dimnames(contrasts(sportscar.ml2$seat))[[2]] <- levels(sportscar.ml2$seat)[1:2]

# Create character vector of "n" for every independent variable in the model, which will let the hierachical model know to assume a random
my_rpar <- rep("n", length(sportscar_model4$coef))
names(my_rpar) <- names(sportscar_model4$coef)
```

##### **Fit a hierarchical model that assumes a heterogeneity of preferences, and correlations between variables**

``` r
# fit model with the assumption that there may be correlations between variables
sportscar_model6 <- mlogit(choice ~ 0 + seat + trans + convert + price + price:segment, data = sportscar.ml , panel = TRUE, rpar = my_rpar, correlation = TRUE)
```

This time, we add the following arguments to the mlogit function:

-   panel: We specify this to be TRUE, as we want to use panel techniques to specify that each simulated respondent's decisions will be measured multiple times
-   rpar: We set rpar equal to our custom variable "my\_rpar," which is a vector coded with 'n' for every coefficient in the model, to specify normally distributed random parameters
-   correlation = We set this to TRUE to take into account any correlations that may exist between independent variables in the model

Now let's look at the result of the revised model:

``` r
## Generate model summary
summary(sportscar_model6)
```

    ## 
    ## Call:
    ## mlogit(formula = choice ~ 0 + seat + trans + convert + price + 
    ##     price:segment, data = sportscar.ml, rpar = my_rpar, correlation = TRUE, 
    ##     panel = TRUE)
    ## 
    ## Frequencies of alternatives:
    ##     1     2     3 
    ## 0.328 0.327 0.345 
    ## 
    ## bfgs method
    ## 40 iterations, 0h:0m:42s 
    ## g'(-H)^-1g = 4.97E-07 
    ## gradient close to zero 
    ## 
    ## Coefficients :
    ##                                              Estimate Std. Error  z-value
    ## seat4                                      -0.0272847  0.0886525  -0.3078
    ## seat5                                       0.5378794  0.0928816   5.7910
    ## transmanual                                -1.6919426  0.0985508 -17.1682
    ## convertyes                                  0.3833417  0.0795414   4.8194
    ## price                                      -0.2899051  0.0172100 -16.8451
    ## price:segmentfun                            0.1481165  0.0372094   3.9806
    ## price:segmentracer                          0.2405219  0.1007354   2.3877
    ## chol.seat4:seat4                           -0.0146195  0.1280270  -0.1142
    ## chol.seat4:seat5                           -0.2693098  0.1412984  -1.9060
    ## chol.seat5:seat5                            0.5985816  0.1411732   4.2401
    ## chol.seat4:transmanual                      1.3107934  0.1557977   8.4134
    ## chol.seat5:transmanual                      0.1733163  0.1418299   1.2220
    ## chol.transmanual:transmanual               -0.8415295  0.1581268  -5.3219
    ## chol.seat4:convertyes                      -0.6235723  0.1315881  -4.7388
    ## chol.seat5:convertyes                       0.2821136  0.1302368   2.1662
    ## chol.transmanual:convertyes                 0.6687938  0.1581924   4.2277
    ## chol.convertyes:convertyes                  0.8541375  0.1525179   5.6002
    ## chol.seat4:price                            0.0194539  0.0277145   0.7019
    ## chol.seat5:price                           -0.0326766  0.0228031  -1.4330
    ## chol.transmanual:price                     -0.0681851  0.0292626  -2.3301
    ## chol.convertyes:price                       0.0032166  0.0238410   0.1349
    ## chol.price:price                           -0.0039220  0.0451518  -0.0869
    ## chol.seat4:price:segmentfun                 0.0653759  0.0517251   1.2639
    ## chol.seat5:price:segmentfun                 0.0297203  0.0435795   0.6820
    ## chol.transmanual:price:segmentfun          -0.0047863  0.0610705  -0.0784
    ## chol.convertyes:price:segmentfun           -0.0100005  0.0514776  -0.1943
    ## chol.price:price:segmentfun                 0.0374996  0.0727322   0.5156
    ## chol.price:segmentfun:price:segmentfun      0.0191037  0.0699075   0.2733
    ## chol.seat4:price:segmentracer              -0.0752970  0.0665363  -1.1317
    ## chol.seat5:price:segmentracer               0.0055760  0.0575831   0.0968
    ## chol.transmanual:price:segmentracer         0.0880834  0.0662303   1.3300
    ## chol.convertyes:price:segmentracer          0.0198694  0.0615339   0.3229
    ## chol.price:price:segmentracer               0.0026779  0.0813209   0.0329
    ## chol.price:segmentfun:price:segmentracer   -0.0044976  0.0626892  -0.0717
    ## chol.price:segmentracer:price:segmentracer  0.0009232  0.0640640   0.0144
    ##                                             Pr(>|z|)    
    ## seat4                                        0.75826    
    ## seat5                                      6.996e-09 ***
    ## transmanual                                < 2.2e-16 ***
    ## convertyes                                 1.440e-06 ***
    ## price                                      < 2.2e-16 ***
    ## price:segmentfun                           6.874e-05 ***
    ## price:segmentracer                           0.01696 *  
    ## chol.seat4:seat4                             0.90909    
    ## chol.seat4:seat5                             0.05665 .  
    ## chol.seat5:seat5                           2.235e-05 ***
    ## chol.seat4:transmanual                     < 2.2e-16 ***
    ## chol.seat5:transmanual                       0.22171    
    ## chol.transmanual:transmanual               1.027e-07 ***
    ## chol.seat4:convertyes                      2.150e-06 ***
    ## chol.seat5:convertyes                        0.03030 *  
    ## chol.transmanual:convertyes                2.361e-05 ***
    ## chol.convertyes:convertyes                 2.140e-08 ***
    ## chol.seat4:price                             0.48272    
    ## chol.seat5:price                             0.15186    
    ## chol.transmanual:price                       0.01980 *  
    ## chol.convertyes:price                        0.89268    
    ## chol.price:price                             0.93078    
    ## chol.seat4:price:segmentfun                  0.20626    
    ## chol.seat5:price:segmentfun                  0.49525    
    ## chol.transmanual:price:segmentfun            0.93753    
    ## chol.convertyes:price:segmentfun             0.84597    
    ## chol.price:price:segmentfun                  0.60614    
    ## chol.price:segmentfun:price:segmentfun       0.78464    
    ## chol.seat4:price:segmentracer                0.25777    
    ## chol.seat5:price:segmentracer                0.92286    
    ## chol.transmanual:price:segmentracer          0.18353    
    ## chol.convertyes:price:segmentracer           0.74677    
    ## chol.price:price:segmentracer                0.97373    
    ## chol.price:segmentfun:price:segmentracer     0.94281    
    ## chol.price:segmentracer:price:segmentracer   0.98850    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Log-Likelihood: -1571
    ## 
    ## random coefficients
    ##                    Min.     1st Qu.      Median        Mean     3rd Qu.
    ## seat4              -Inf -0.03714540 -0.02728473 -0.02728473 -0.01742405
    ## seat5              -Inf  0.09516142  0.53787941  0.53787941  0.98059740
    ## transmanual        -Inf -2.74906192 -1.69194263 -1.69194263 -0.63482334
    ## convertyes         -Inf -0.48181237  0.38334174  0.38334174  1.24849586
    ## price              -Inf -0.34267566 -0.28990510 -0.28990510 -0.23713454
    ## price:segmentfun   -Inf  0.09147787  0.14811650  0.14811650  0.20475513
    ## price:segmentracer -Inf  0.16105097  0.24052194  0.24052194  0.31999292
    ##                    Max.
    ## seat4               Inf
    ## seat5               Inf
    ## transmanual         Inf
    ## convertyes          Inf
    ## price               Inf
    ## price:segmentfun    Inf
    ## price:segmentracer  Inf

##### **Revised Model Results**

-   Seat effects:
    -   5-seat vehicles were once again statistically more likely than 2-seaters to be chosen, while there was a negligible difference between that of 4-seaters and 2-seaters.
-   Transmissions:
    -   There was still a statistically significant negative effect of manual transmissions on whether or not a sportscar was chosen.
-   Convertible roofs:
    -   Sportscars with convertible tops were significantly more likely to be chosen than those with standard roofs
-   Price:
    -   Price still had a very strong effect on whether or not a vehicle was chosen, with lower priced sportcars typically chosen over pricier ones
-   Price+Segment:
    -   Sportscars in the fun segment were statistically more likely to be chosen at higher price points than the basic model at the 99.9% confidence level
    -   There was a weaker effect, but with 95% confidence, we can say that racer sportscars were statistically more likely than basic sportscars to be chosen at higher price points.

#### **Create a share prediction function for the hierachical model and generate a random draw of 1,000 decisions**

Now let's go back to our hypothetical scenario. Let's run a simulation in which respondents were faced 1,000 times with the choice between our client's 2-seat automatic convertible racer and the 2=seat racers of the other two competitors. What would've likely happened?

``` r
## Build share prediction function
coef_means <- sportscar_model6$coef[1:7]
Sigma <- cov.mlogit(sportscar_model6)

model6_coded <- model.matrix(update(sportscar_model6$formula, 0 ~ .), data = prod)[,-1]

share <- matrix(NA, nrow=1000, ncol=nrow(model6_coded))

# Compute a random draw of 1,000 buyers
for (i in 1:1000) { 
  # Draw a coefficient vector from the normal distribution
  coef <- mvrnorm(1, mu=coef_means, Sigma=Sigma)
  # Compute utilities for those coef
  utility <- model6_coded %*% coef
  # Compute probabilites according to logit formuila
  share[i,] <- exp(utility) / sum(exp(utility))
}  

head(share)
```

    ##           [,1]       [,2]      [,3]
    ## [1,] 0.6783683 0.07291334 0.2487184
    ## [2,] 0.6470350 0.03501524 0.3179498
    ## [3,] 0.2495950 0.24789639 0.5025086
    ## [4,] 0.6516135 0.10344097 0.2449456
    ## [5,] 0.2375664 0.05284192 0.7095917
    ## [6,] 0.2435034 0.25589643 0.5006001

### **Plot final choice share based on mean share of 1,000 simulated responses**

    ##   colMeans(share) car seat  trans convert price segment
    ## 1       0.5223137   1    2   auto     yes    30   racer
    ## 2       0.1252505   2    2 manual     yes    30   racer
    ## 3       0.3524358   3    2   auto      no    30   racer

![](Choice_Modeling_Sports_Car_Sales_files/figure-markdown_github/plot-1.png)

In the final model, our client's proposed car takes a commanding choice share, with 52% of the simulated responses. Sounds like our client is going to be getting into the 2-seat sportscar market soon!

By tweaking variables in the prod dataframe, the models presented here would give great intel for car manufacturers and dealers alike. Going further, if we had demographic data on the respondents, one could use clustering techniques to do a market segmentation to further see what types of consumers would be most amenable to different vehicle features.

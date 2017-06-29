---
layout: post
title: Day 1
---











## Modeling differences between two groups using Pystan

Ara Winter<br>
2017-06-29

This documents uses a model that compares the means and standard deviations of two groups. This is carried out in the pystan framework. 


```python
import pystan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_context('notebook')
from IPython.display import Image
%matplotlib inline
```

### Example: Three centuries of English grain production
To demostrate this model we will be using the Three centuries of English crop yields 1211-1491. The full dataset is describe here: http://www.cropyields.ac.uk/login.php . We are using a small subset of the total data. The data set contains: wheat gross yield per seed ratio, county, manor, estate, and year. 


#### Reference
Bruce M. S. Campbell (2007), Three centuries of English crops yields, 1211‑1491 [WWW document]. URL http://www.cropyields.ac.uk [accessed on 29/June/2017]

### Where are we?
We are going to look at just two manors in two seperate counties in England. Berkshire and Hampshire and the manors Alresford and Brightwell, respectively. 

<tr>
<td> Berkshire<img src="bioinfonm.github.io/_posts/pystan_musings_part1_img//200px-Berkshire_UK_locator_map_2010.svg.png" title="Berkshire" style="width: 250px;"/> </td>
<td> Hampshire<img src="bioinfonm.github.io/_posts/pystan_musings_part1_img//200px-Hampshire_UK_locator_map_2010.svg.png" alt="Hampshire" style="width: 250px;"/> </td>
</tr>



maps by Unknown (wikimedia), distributed Creative Commons Attribution-Share Alike 3.0 Unported

### Data
First we need to load the .csv with the data in it. Here we are going to use pandas (acts like a R data frame) and from there pull out the data we need.


```python
# Import data
grain_data = pd.read_csv('./three_centuries_of_grain/berkshire_hampshire_grain_1349_1458.txt', sep="\t")
# Here I am telling pd.read_csv that in the folder where my notebook is I have another folder called:
# three_centuries_of_grain. And in that folder I have a tab (\t) seperated comma seperated file. 
# We then ask pandas to read it in and stick it in a data frame called grain_data
```


```python
grain_data.head(4)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>County</th>
      <th>Manor</th>
      <th>Estate</th>
      <th>start</th>
      <th>end</th>
      <th>wheat_gross_yield_per_seed_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Hampshire</td>
      <td>Alresford</td>
      <td>Bishop of Winchester</td>
      <td>1349</td>
      <td>1350</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hampshire</td>
      <td>Alresford</td>
      <td>Bishop of Winchester</td>
      <td>1350</td>
      <td>1351</td>
      <td>3.09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hampshire</td>
      <td>Alresford</td>
      <td>Bishop of Winchester</td>
      <td>1351</td>
      <td>1352</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hampshire</td>
      <td>Alresford</td>
      <td>Bishop of Winchester</td>
      <td>1352</td>
      <td>1353</td>
      <td>3.17</td>
    </tr>
  </tbody>
</table>
</div>




```python
grain_data.tail(4)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>County</th>
      <th>Manor</th>
      <th>Estate</th>
      <th>start</th>
      <th>end</th>
      <th>wheat_gross_yield_per_seed_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>214</th>
      <td>Berkshire</td>
      <td>Brightwell</td>
      <td>Bishop of Winchester</td>
      <td>1454</td>
      <td>1455</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>215</th>
      <td>Berkshire</td>
      <td>Brightwell</td>
      <td>Bishop of Winchester</td>
      <td>1455</td>
      <td>1456</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>216</th>
      <td>Berkshire</td>
      <td>Brightwell</td>
      <td>Bishop of Winchester</td>
      <td>1456</td>
      <td>1457</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>217</th>
      <td>Berkshire</td>
      <td>Brightwell</td>
      <td>Bishop of Winchester</td>
      <td>1457</td>
      <td>1458</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
grain_data.describe()
# Most of this doesn't matter execept for the wheat gross yield per seed ratio.
# The mean and std can be helpful for setting our priors.
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start</th>
      <th>end</th>
      <th>wheat_gross_yield_per_seed_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>218.00000</td>
      <td>218.00000</td>
      <td>129.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1403.00000</td>
      <td>1404.00000</td>
      <td>4.420853</td>
    </tr>
    <tr>
      <th>std</th>
      <td>31.53668</td>
      <td>31.53668</td>
      <td>1.607587</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1349.00000</td>
      <td>1350.00000</td>
      <td>0.970000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1376.00000</td>
      <td>1377.00000</td>
      <td>3.170000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1403.00000</td>
      <td>1404.00000</td>
      <td>4.150000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1430.00000</td>
      <td>1431.00000</td>
      <td>5.560000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1457.00000</td>
      <td>1458.00000</td>
      <td>8.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# You will notice we have lots of missing data.
grain_data.isnull().sum()
```




    County                               0
    Manor                                0
    Estate                               0
    start                                0
    end                                  0
    wheat_gross_yield_per_seed_ratio    89
    dtype: int64



### Exploratory plots


```python
# Here we just want to plot the data over time. 
sns.lmplot("start", "wheat_gross_yield_per_seed_ratio", data=grain_data, hue='County', fit_reg=True)
```




    <seaborn.axisgrid.FacetGrid at 0x11e53dc50>




![png](bioinfonm.github.io/_posts/pystan_musings_part1_img/output_15_1.png)



```python
# It's likely that a non-linear fit to the data is better but that's a different question and different model.
sns.lmplot("start", "wheat_gross_yield_per_seed_ratio", data=grain_data, hue='County', lowess=True)
```




    <seaborn.axisgrid.FacetGrid at 0x12008fb70>




![png](/pystan_musings_part1_img/output_16_1.png)



```python
# And here we just look at the bulk difference between Hampsire and Berkshire
sns.stripplot(x="County", y="wheat_gross_yield_per_seed_ratio", data=grain_data);
```


![png](/pystan_musings_part1_img/output_17_0.png)


### Our question
Just by looking at the data we can dream of lots of questions. There are two that I am interested in:<br>
Is the wheat gross yield per seed ratio different between Hampsire and Berkshire from 1349-1457?<br>
When is the wheat gross yield per seed ratio different between Hampsire and Berkshire from 1349-1457?<br>

The second one is arguable more interesting. And ideally we would dig into what things predict the yield from year to year. Is it the previous years yield or spring plantings or something else?

In this notebook we are justing going to address the first question. 

### Model
We have lots of choices for modeling. Do we only use complete cases? This will reduce the size of the data. Or do we try and model it with NAs. I am not sure of the NA's are missing data or actually zero for wheat gross yield per seed ratio.<br>
The elements of a Stan model is best explained by the Stan folks:
http://m-clark.github.io/workshops/bayesian/04_R.html<br>
First I am going to split my dataframe by County


```python
# This allows us to only have complete cases. 
matched_grain = grain_data.dropna(axis=0, how='any')
```


```python
# Just checking to make sure all the NA's are gone.
matched_grain.isnull().sum()
```




    County                              0
    Manor                               0
    Estate                              0
    start                               0
    end                                 0
    wheat_gross_yield_per_seed_ratio    0
    dtype: int64




```python
matched_grain.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start</th>
      <th>end</th>
      <th>wheat_gross_yield_per_seed_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>129.000000</td>
      <td>129.000000</td>
      <td>129.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1389.193798</td>
      <td>1390.193798</td>
      <td>4.420853</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25.945278</td>
      <td>25.945278</td>
      <td>1.607587</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1349.000000</td>
      <td>1350.000000</td>
      <td>0.970000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1367.000000</td>
      <td>1368.000000</td>
      <td>3.170000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1386.000000</td>
      <td>1387.000000</td>
      <td>4.150000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1411.000000</td>
      <td>1412.000000</td>
      <td>5.560000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1447.000000</td>
      <td>1448.000000</td>
      <td>8.800000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This doesn't look to much different from our earlier raw plots. So the complete cases data should 
# be good to work with.
sns.lmplot("start", "wheat_gross_yield_per_seed_ratio", data=matched_grain, hue='County', lowess=True)
```




    <seaborn.axisgrid.FacetGrid at 0x12088df98>




![png](/pystan_musings_part1_img/output_23_1.png)



```python
# We seperate out each county into it's own data frame. 
berkshire = matched_grain.loc[matched_grain['County'] == 'Berkshire']
hampshire = matched_grain.loc[matched_grain['County'] == 'Hampshire']
```


```python
# Here is the data to model. Note we convert the dataframe column to a python list. 
y1 = berkshire.wheat_gross_yield_per_seed_ratio.tolist()
y2 = hampshire.wheat_gross_yield_per_seed_ratio.tolist()
# Here we just ask what is the length of the data.
N1 = len(y1)
N2 = len(y2)
```


```python
# Dictionary containing all data to be passed to STAN
compare_groups = {'y1': y1,
 'y2': y2,
 'n1': N1,'n2':N2}
```


```python
# The Stan model as a string.
model_string = """
data { //The data Stan should be expecting from the data list.
  int n1; //sample size
  int n2; //sample size
  vector[n1] y1; //type, dimension, name of variable. Group y1
  vector[n2] y2; //type, dimension, name of variable. Group y2
}

parameters { //The primary parameters of interest that are to be estimated. 
  real mu1; // mean of y1
  real mu2; // mean of y2
  real<lower=0> sigma1; // standard deviation of y1
  real<lower=0> sigma2; // standard deviation of y2
}

model { // Where your priors and likelihood are specified. Uniform, cauchy, and normal 
        // priors might be a good place to start?
  mu1 ~ uniform(0, 30); // uniform prior, maybe try half-normal, exp, or half-cauchy
  mu2 ~ uniform(0, 30); //
  sigma1 ~ cauchy(0, 20); // standard deviation is using a cauchy, maybe half-cauchy?
  sigma2 ~ cauchy(0, 20);
  y1 ~ normal(mu1, sigma1); // group y1 is using a normal based on the mean and standard deviation
  y2 ~ normal(mu2, sigma2); 
}

generated quantities {
}
"""
```


```python
# Compiling and producing posterior samples from the model.
fit_compare_groups = pystan.stan(model_code=model_string, data=compare_groups,
                  iter=1000, chains=4)
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_7e077d18a2d7b04561975e1153dcea49 NOW.



```python
fit_compare_groups
```




    Inference for Stan model: anon_model_7e077d18a2d7b04561975e1153dcea49.
    4 chains, each with iter=1000; warmup=500; thin=1; 
    post-warmup draws per chain=500, total post-warmup draws=2000.
    
             mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
    mu1      5.45  3.4e-3   0.15   5.14   5.35   5.45   5.55   5.76   2000    1.0
    mu2      3.08  2.4e-3   0.11   2.87   3.01   3.08   3.16    3.3   2000    1.0
    sigma1   1.32  2.6e-3   0.12   1.12   1.24   1.31   1.39   1.56   2000    1.0
    sigma2   0.79  1.7e-3   0.08   0.67   0.74   0.79   0.84   0.96   2000    1.0
    lp__    -70.3    0.04   1.41  -73.7 -71.05 -70.01 -69.21 -68.54   1089    1.0
    
    Samples were drawn using NUTS at Thu Jun 29 12:26:42 2017.
    For each parameter, n_eff is a crude measure of effective sample size,
    and Rhat is the potential scale reduction factor on split chains (at 
    convergence, Rhat=1).




```python
# Plotting the posterior distribution
fit_compare_groups.plot()
```




![png](/pystan_musings_part1_img/output_30_0.png)




![png](/pystan_musings_part1_img/output_30_1.png)



```python
# Instead, show a traceplot for single parameter
fit_compare_groups.plot(['mu1'])
plt.show()
```


![png](/pystan_musings_part1_img/output_31_0.png)



```python
# Wrangles the data from fit_compare_groups into a pandas dataframe
samples = fit_compare_groups.extract(permuted=True)
params = pd.DataFrame({'mu1': samples['mu1'], 'mu2': samples['mu2'],
                       'sigma1': samples['sigma1'],'sigma2': samples['sigma2'],'lp__': samples['lp__']})
```


```python
# Inspect our data frame
params.head(4)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lp__</th>
      <th>mu1</th>
      <th>mu2</th>
      <th>sigma1</th>
      <th>sigma2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-69.623487</td>
      <td>5.694688</td>
      <td>3.088306</td>
      <td>1.288156</td>
      <td>0.769257</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-72.383754</td>
      <td>5.843168</td>
      <td>3.035218</td>
      <td>1.501718</td>
      <td>0.783125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-69.650684</td>
      <td>5.478343</td>
      <td>2.911267</td>
      <td>1.323245</td>
      <td>0.794895</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-73.412380</td>
      <td>5.803150</td>
      <td>3.192793</td>
      <td>1.540021</td>
      <td>0.708185</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting distribution of the difference between mu1 and mu2. In other words the difference in the means.
sns.distplot((params.mu1 - params.mu2), kde=False, rug=False);
# What does this range mean?
```


![png](/pystan_musings_part1_img/output_34_0.png)



```python
# The probability that mu1 is smaller than mu2
# The probabilty that the mean of Berkshire wheat production is smaller then the mean of Hampshire wheat production.
l = params.mu1 < params.mu2
sum(l)/len(l)
```




    0.0




```python
# Plotting distribution of the difference between sigma1 and sigma2. In other words the difference in the 
# standard deviations.
sns.distplot((params.sigma1 - params.sigma2), kde=False, rug=False);
# What does this range mean?
```


![png](/pystan_musings_part1_img/output_36_0.png)



```python
l = params.sigma1 < params.sigma2
sum(l)/len(l)
```




    0.0



### Conclusions
This may seem like a lot of work to compare two groups of things. But it unmasks the model that we usally see as a one liner, like the t-test or something similar. <br>
We can also say something about grain production between two counties in England from 1348 - 1448. The mean grain output from Hampshire is greater then the output from Berkshire. However we are left with other questions better left for different models.

### References
Gelman A, Lee D, Guo J. Stan: A probabilistic programming language for Bayesian inference and optimization. Journal of Educational and Behavioral Statistics. 2015 Oct;40(5):530-43.<br>
Breakdown of the Stan model http://m-clark.github.io/workshops/bayesian/03_stan.html <br>
Prior Choice Recommendations https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations <br>


### Better practices
Two things that stand out. We should have heavily annotated priors. So why did I choose mu1 ~ uniform(0, 30) or sigma1 ~ cauchy(0, 20). Once I get a better handle on what does numbers do I'll come back an do an annotated prior entry.

In the data section of our Stan model we really should have bounds on our data that serves as checks. 
So <i>int n1</i> becomes <i>int&lt;lower=1> n1</i>. This sets a boundary that we expect at least one sample.

### Future work
Seaborn handled the linear models for us. So the next project has three parts:<br>
Simple linear model with time predicting grain production<br>
Simple non-linear model fitting of time predicting grain production<br>
Can we write a model that let's us know if previous years harvest influcence the next year<br>

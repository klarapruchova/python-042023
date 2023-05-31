import pandas
import seaborn
import matplotlib.pyplot as plt
from scipy import stats
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pandas.read_csv("Life-Expectancy-Data-Updated.csv")
data_2015 = data[data["Year"] == 2015]
#seaborn.regplot(data_2015, x="GDP_per_capita", y="Life_expectancy", scatter_kws={"s": 1}, line_kws={"color":"r"})
#plt.show()

formula = "Life_expectancy ~ GDP_per_capita"
mod = smf.ols(formula=formula, data=data_2015)
res = mod.fit()
#print(res.summary())

# H0 = rezidua mají normální rozdělení
# H1 = rezidua nemají normální rozdělení
formula2 = "Life_expectancy ~ GDP_per_capita + Schooling + Incidents_HIV + Diphtheria + Polio + BMI + Measles"
mod = smf.ols(formula=formula2, data=data_2015)
res = mod.fit()
#print(res.summary())
# obě p-hodnoty jsou > 0.05, nelze tedy zamítnout H0 o normálním rozdělením reziduí
# koeficient determinace R-squared měl u prvního modelu hodnotu 0,396, nyní má hodnotu 0,790 => nový model je přesnější

formula3 = "Life_expectancy ~ GDP_per_capita + Schooling + Incidents_HIV + Polio + BMI + Measles"
mod = smf.ols(formula=formula3, data=data_2015)
res = mod.fit()
print(res.summary())
# nejvyšší p-hodnotu měl řádek proměnné "Diphtheria", po jeho odstranění z modelu se ostatní p-hodnoty snížily => jsou tedy statisticky významnější
# koeficient determinace zůstal stejný => kvalita modelu je stejná




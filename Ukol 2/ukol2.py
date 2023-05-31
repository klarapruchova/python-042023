import pandas
from scipy import stats
import matplotlib.pyplot as plt

# 1) Inflace
# H0: procenta lidí, kteří považují inflaci a růst životních nákladů za jeden ze svých nejzávažnějších problémů mají normální rozdělení
# H1: procenta lidí, kteří považují inflaci a růst životních nákladů za jeden ze svých nejzávažnějších problémů nemají normální rozdělení
data = pandas.read_csv("ukol_02_a.csv")
res1 = stats.shapiro(data["97"])
# statistic=0.9694532752037048, pvalue=0.33090925216674805 => H0 nezamítáme, data mají normální rozdělení
res2 = stats.shapiro(data["98"])
# statistic=0.9803104996681213, pvalue=0.687289297580719 => H0 nezamítáme, data mají normální rozdělení

# Ověř, zda se procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, změnilo.
# H0: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, se nezměnilo.
# H1: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, se změnilo.
# Protože porovnávám stejnou skupinu států ve dvou různých časových obdobích, využiju párový t-test, který předpokládá normální rozdělení.
res = stats.ttest_rel(data["97"], data["98"])
# statistic=3.868878598419143, pvalue=0.0003938172257904746, df=40 => p-hodnota < 0.05, zamítáme nulovou hypotézu a platí, že rocento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, se změnilo

# 2) Důvěra ve stát a v EU
data2 = pandas.read_csv("ukol_02_b.csv")
data_EU = pandas.read_csv("countries.csv")
# Zajímají mě jen data pro státy EU:
data_2EU = pandas.merge(data2, data_EU, on=["Country"])

# H0: Procenta lidí, kteří věří své národní vládě mají normální rozdělení.
# H1: Procenta lidí, kteří věří své národní vládě nemají normální rozdělení.
res3 = stats.shapiro(data_2EU["National Government Trust"])
# statistic=0.9438267350196838, pvalue=0.15140558779239655 => H0 nezamítáme, data mají normální rozdělení

# H0: Procenta lidí, kteří věří EU mají normální rozdělení.
# H1: Procenta lidí, kteří věří EU nemají normální rozdělení.
res4 = stats.shapiro(data_2EU["EU Trust"])
# statistic=0.9735807180404663, pvalue=0.6981646418571472 => H0 nezamítáme, data mají normální rozdělení

# Existuje korelace mezi procentem lidí, které věří EU v každé ze zemí, a procentem lidí, kteří EU nevěří?
# H0: Procento lidí, kteří věří své národní vládě a procento lidí, kteří věří EU, nejsou statisticky závislé.
# H1: Procento lidí, kteří věří své národní vládě a procento lidí, kteří věří EU, jsou statisticky závislé.
# Data mají normální rozdělení, použiju test založený na Pearsonově korelačním koeficientu.
res5 = stats.pearsonr(data_2EU["National Government Trust"], data_2EU["EU Trust"])
# statistic=0.6097186340024556, pvalue=0.0007345896228823406 => zamítám H0, korelace mezi daty existuje

# 3) Důvěra v EU a euro
data_EURO = data_2EU[data_2EU["Euro"] == 1]
data_nemajiEURO = data_2EU[data_2EU["Euro"] == 0]
# Důvěřují EU více lidé, kteří žijí ve státech platící eurem?
# H0: Důvěra lidí v EU je stejná v zemích platících eurem i v zemích mimo eurozónu.
# H1: Důvěra lidí v EU je různá v zemích platících eurem i v zemích mimo eurozónu.
# Předpokládáme normální rozdělení dat. Použijeme nepárový t-test
res6 = stats.ttest_ind(data_EURO["EU Trust"], data_nemajiEURO["EU Trust"])
# statistic=-0.33471431258258433, pvalue=0.7406326832748829 => nelze zamítnout H0, platí, že důvěra lidí v EU je stejná ve státech platících eurem i mimo eurozónu.
print(res6)


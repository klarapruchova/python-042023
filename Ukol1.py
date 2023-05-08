import pandas
import numpy
import matplotlib.pyplot as plt

data = pandas.read_csv("1976-2020-president.csv")
# 1) Urči pořadí jednotlivých kandidátů v jednotlivých státech a v jednotlivých letech (pomocí metody rank()).
data["rank"] = data.groupby(["year", "state"])["candidatevotes"].rank(ascending=False)

# 2) Vytvoř novou tabulku, která bude obsahovat pouze vítěze voleb.
data_vitezove = data[data["rank"] == 1]

# 3) Pomocí metody shift() přidej nový sloupec, abys v jednotlivých řádcích měl(a) po sobě vítězné strany ve dvou po sobě jdoucích letech.
data_vitezove["previous_winner_party"] = data_vitezove.groupby(["state"])["party_simplified"].shift(1)
data_vitezove = data_vitezove.sort_values(["state", "year"])

# 4) Porovnej, jestli se ve dvou po sobě jdoucích letech změnila vítězná strana.
data_vitezove["change"] = numpy.where(data_vitezove["party_simplified"] == data_vitezove["previous_winner_party"], 0, 1)
data_vitezove_od1980 = data_vitezove[data_vitezove["year"] != 1976]

# 5) Proveď agregaci podle názvu státu a seřaď státy podle počtu změn vítězných stran.
data_vitezove_pivot = data_vitezove_od1980.groupby(["state"])["change"].sum()
data_vitezove_pivot = pandas.DataFrame(data_vitezove_pivot)
data_vitezove_pivot = data_vitezove_pivot.sort_values("change", ascending=False)

# 6) Vytvoř sloupcový graf s 10 státy, kde došlo k nejčastější změně vítězné strany. Jako výšku sloupce nastav počet změn.
data_vitezove_pivot = data_vitezove_pivot.iloc[:10]
data_vitezove_pivot = data_vitezove_pivot[data_vitezove_pivot["change"] >= 4]
data_vitezove_pivot["change"].plot(kind="bar")
plt.legend(["change"])
plt.xlabel("state")
#plt.show()
#print(data_vitezove_pivot.head(20))

# II. část 1) Přidej do tabulky sloupec, který obsahuje absolutní rozdíl mezi vítězem a druhým v pořadí.
data_margin = data_vitezove.sort_values(["year"])
data_margin["second_candidate_votes"] = data.groupby(["year"])["candidatevotes"].shift(-1)
data_margin["margin"] = data_margin["candidatevotes"] - data_margin["second_candidate_votes"]

# II. část 2) Přidej sloupec s relativním marginem, tj. rozdílem vyděleným počtem hlasů.
data_margin["relative_margin"] = data_margin["margin"] / data_margin["totalvotes"]

# II. část 3) Seřaď tabulku podle velikosti relativního marginu a zjisti, kdy a ve kterém státě byl výsledek voleb nejtěsnější.
data_margin = data_margin.sort_values(["relative_margin"])
#print(data_margin)

#II. čáast 4) Vytvoř pivot tabulku, která zobrazí pro jednotlivé volební roky, kolik států přešlo od Republikánské strany k Demokratické straně, kolik států přešlo od Demokratické strany k Republikánské straně a kolik států volilo kandidáta stejné strany.
def swing(row):
    if row["change"] == 1 and row["party_simplified"] == "REPUBLICAN":
        return "to Rep."
    if row["change"] == 1 and row ["party_simplified"] == "DEMOCRAT":
        return "to Dem."
    else:
        return "no swing"

data_vitezove_od1980["swing"] = data_vitezove_od1980.apply(swing, axis=1)
data_pivot_swing = pandas.pivot_table(data_vitezove_od1980, values="change", index="year", columns="swing", aggfunc=len)

print(data_pivot_swing.head(20))
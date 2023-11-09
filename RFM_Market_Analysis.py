########## Load Library  ########

import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import datetime as dt

# pip install apriori
# pip install mlxtend

import xlrd
import re
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 1000)

##### Read datasets and merge #######

df1 = pd.read_csv("QUANTIUM/QVI_purchase_behaviour.csv")

df1.columns

df2 = pd.read_excel("QUANTIUM/QVI_transaction_data.xlsx")

df2.columns

dff = pd.merge(
    df1, df2, how="inner", left_on="LYLTY_CARD_NBR", right_on="LYLTY_CARD_NBR"
)

dff.columns

df = dff.copy()

###################             EDA             ###############################


def Check_df(dataframe, head=5):
    print("####### SHAPE ###########")
    print(dataframe.shape)
    print("############# HEAD #######")
    print(dataframe.head(head))
    print("########### TAIL ##########")
    print(dataframe.tail(head))
    print("############ NaN ###########")
    print(dataframe.isnull().sum())
    print("############ Quantiles ######")
    print(dataframe.quantile([0, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.99, 1]).T)
    print("########### TYPE ##############")
    print(dataframe.dtypes)


Check_df(df)

######## Cleansing and Tunning #############


### Check Duplicate and Cleansing #######

df.duplicated().sum()

duplicates = df[df.duplicated()]

print(duplicates)

df.shape

df.drop_duplicates(inplace=True)

Check_df(df)

###########  OUTLIER   ######################


def grab_col_names(dataframe, cat_th=5, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

###### Date Problem Solving ######

cat_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
print(cat_cols)
num_cols = [col for col in num_cols if col not in cat_cols]

###### Outlier ######################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[
        (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
    ].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


df.head(20)
df.describe()


############# Feature Engineering #############

# Add a new column to data with packet sizes and extract sizes from product name column

###### Pack_size insert ##########

df.insert(8, "PACK_SIZE", df["PROD_NAME"].str.extract(r"(\d+)").astype(float), True)
df.head()
# Sort by packet sizes to check for outliers

df_sort = df.sort_values(by="PACK_SIZE")
df_sort.head()
df_sort.tail()
df_sort.describe().T

###### Histogram ##########

# Minimum packet size is 70g while max is 380g - this is reasonable.
# Plot a histogram to visualise distribution of pack sizes.
plt.hist(df["PACK_SIZE"], weights=df["PROD_QTY"])
plt.xlabel("Packet size (g)")
plt.ylabel("Quantity")
plt.show()

########## Brand Name insert ######

# Now that the pack size looks reasonable, we can create the brand names using the first word of each product name.

# Add a column to extract the first word of each product name to.

df.insert(9, "BRAND_NAME", df["PROD_NAME"].str.split().str.get(0), True)

Check_df(df)

# Then print all unique entries to check the brand names created

df["BRAND_NAME"].unique()

df["BRAND_NAME"].nunique()
df.columns
df["PROD_NAME"].unique()

# Natural, NNC, Natural Chip Co
# Red Rock Deli, RRD,Red
# Grain Waves, Grain, GrnWves
# Sunbites', 'Snbts'
#'Infuzions',  'Infzns'
#'Doritos', 'Dorito'
#'WW', 'Woolworths'
#'Smiths','Smith'


# Create a function to identify the string replacements needed.


def replace_name(dataframe):
    name = dataframe["BRAND_NAME"]
    if name == "Natural":
        return "Natural Chip Co"
    elif name == "NCC":
        return "Natural Chip Co"
    elif name == "RRD":
        return "Red Rock Deli"
    elif name == "Red":
        return "Red Rock Deli"
    elif name == "Grain":
        return "Grain Waves"
    elif name == "GrnWves":
        return "Grain Waves"
    elif name == "Snbts":
        return "Sunbites"
    elif name == "Infzns":
        return "Infuzions"
    elif name == "Dorito":
        return "Doritos"
    elif name == "WW":
        return "Woolworths"
    elif name == "Smith":
        return "Smiths"
    else:
        return name


# Then apply the function to clean the brand names

df["BRAND_NAME"] = df.apply(lambda dataframe: replace_name(dataframe), axis=1)

# Check that there are no duplicate brands

df["BRAND_NAME"].unique()
df["BRAND_NAME"].nunique()

df.shape

# Check for nulls in the full dataset

df.isnull().values.any()

# looks like all the data is reasonable so export to CSV

df.to_csv("QVI_fulldata.csv")

################### Data analysis #################################


"""Who spends the most on chips (total sales), describing customers by lifestage and how premium their general purchasing behaviour is
How many customers are in each segment
How many chips are bought per customer by segment
What's the average chip price by customer segment
Some more information from the data team that we could ask for, to analyse with the chip information for more insight includes

The customer’s total spend over the period and total spend for each transaction to understand what proportion of their grocery spend is on chips.
Spending on other snacks, such as crackers and biscuits, to determine the preference and the purchase frequency of chips compared to other snacks
Proportion of customers in each customer segment overall to compare against the mix of customers who purchase chips
Firstly, we want to take a look at the split of the total sales by LIFESTAGE and PREMIUM_CUSTOMER."""

# calculate total sales by lifestage and member type and generate a list
df.head()
QVI_fulldata = df.groupby(["LIFESTAGE", "PREMIUM_CUSTOMER"], as_index=False)[
    "TOT_SALES"
].agg(["sum"])

QVI_fulldata = QVI_fulldata.rename(columns={"sum": "sum_tot_sales"})

QVI_fulldata.sort_values(by="sum_tot_sales", ascending=False)

####### Get the total sales #########

total_sales = df["TOT_SALES"].agg(["sum"])["sum"]

# Plot a chart of the total sales by lifestage and PREMIUM_CUSTOMER


total_sales_chart = (
    df.groupby(["LIFESTAGE", "PREMIUM_CUSTOMER"], as_index=False)["TOT_SALES"]
    .agg(["sum", "mean"])
    .unstack("PREMIUM_CUSTOMER")
    .fillna(0)
)
ax = total_sales_chart["sum"].plot(kind="barh", stacked=True, figsize=(15, 5))

# Add percentages of the summed total sales as labels to each bar
# .patches is everything inside of the chart

for rect in ax.patches:
    # Find where everything is located
    height = rect.get_height()
    width = rect.get_width()
    label = width / total_sales * 100
    x = rect.get_x()
    y = rect.get_y()

    label_text = f"{(label):.1f}%"

    # Set label positions
    label_x = x + width / 2
    label_y = y + height / 2

    # only plot labels greater than given width
    if width > 0:
        ax.text(label_x, label_y, label_text, ha="center", va="center", fontsize=8)

ax.set_xlabel("Total Sales")
ax.set_title("Total Sales Distributions By Customer Lifestage and Customer Segments")
plt.show()


# Here, we can see the most sales are from
# Older families - Budget, Young singles/couples - Mainstream and Retirees - Mainstream.
# We can see if this is because of the customer numbers in each segment.


# Plot the numbers of customers in each segment by counting the unique LYLTY_CARD_NBR entries

sum_customers = (
    df.groupby(["LIFESTAGE", "PREMIUM_CUSTOMER"])["LYLTY_CARD_NBR"]
    .agg("nunique")
    .unstack("PREMIUM_CUSTOMER")
    .fillna(0)
)
ax = sum_customers.plot(kind="barh", stacked=True, figsize=(15, 5))

# Add customer numbers as labels to each bar
# .patches is everything inside of the chart
for rect in ax.patches:
    # Find where everything is located
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()

    label_text = f"{(width):.0f}"

    # Set label positions
    label_x = x + width / 2
    label_y = y + height / 2

    # only plot labels greater than given width
    if width > 0:
        ax.text(label_x, label_y, label_text, ha="center", va="center", fontsize=8)

ax.set_xlabel("Number of customers")
ax.set_title(
    "Distribution of the Number of Customers By Customer Lifestage and Customer Segments"
)
plt.show()


# Here, we can see the most sales are from Older families - Budget, Young singles/couples - Mainstream and Retirees - Mainstream.
# We can see if this is because of the customer numbers in each segment.


# Plot the average no of chip packets bought per customer by LIFESTAGE and PREMIUM_CUSTOMER.

numbers_packets = df.groupby(["LIFESTAGE", "PREMIUM_CUSTOMER"])[
    "PROD_QTY"
].sum() / df.groupby(["LIFESTAGE", "PREMIUM_CUSTOMER"])["LYLTY_CARD_NBR"].nunique(0)
ax = (
    numbers_packets.unstack("PREMIUM_CUSTOMER")
    .fillna(0)
    .plot.bar(stacked=False, figsize=(10, 5))
)
ax.set_ylabel("Avg no packets purchased")
ax.set_title("Chips Purchased by Customer Lifestage and Customer Segments")
plt.xticks(rotation=45)
plt.show()


# Older families and young families in general buy more chips per customer.
# We can also investigate the average price per unit sold by LIFESTAGE and PREMIUM_CUSTOMER.

# Create a column for the unit price of chips purchased per transaction

df["UNIT_PRICE"] = df["TOT_SALES"] / df["PROD_QTY"]
df.head()


# Plot the distribution of the average unit price per transaction by LIFESTAGE and PREMIUM_CUSTOMER.


avg_unitspriece = (
    df.groupby(["LIFESTAGE", "PREMIUM_CUSTOMER"], as_index=False)["UNIT_PRICE"]
    .agg(["mean"])
    .unstack("PREMIUM_CUSTOMER")
    .fillna(0)
)
ax = avg_unitspriece["mean"].plot.bar(stacked=False, figsize=(10, 5))
ax.set_ylabel("Avg unit price per transaction")
ax.set_title(
    "Average Unit Price Per Transaction by Customer Lifestage and Customer Segments"
)
plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.xticks(rotation=45)
plt.show()


# For young and midage singles/couples, the mainstream group are more willing to pay more for a packet of chips
# than their budget and premium counterpart. Given the total sales, as well as the number of customers buying chips,
# is higher in these groups compared to the non-mainstream groups, this suggests that chips
# may not be the choice of snack for these groups. Further information on shopping habits would be useful in this case.

################ Data-driven look into specific customer segments for insights #######

# We have found quite a few interesting insights that we can dive deeper into.
# We might want to target customer segments that contribute the most to sales to retain them or further increase sales.
# Let's look at Mainstream - young singles/couples.

# Create a visual of what brands young singles/couples are purchasing the most for a general indication

young_mainstream = df.loc[df["LIFESTAGE"] == "YOUNG SINGLES/COUPLES"]
young_mainstream = young_mainstream.loc[
    young_mainstream["PREMIUM_CUSTOMER"] == "Mainstream"
]
ax = (
    young_mainstream["BRAND_NAME"]
    .value_counts()
    .sort_values(ascending=True)
    .plot.barh(figsize=(10, 5))
)
ax.set_xlabel("Numbers of packets purchased")
ax.set_ylabel("Brands")
plt.show()

final = df.copy()
final["group"] = final["LIFESTAGE"] + " - " + final["PREMIUM_CUSTOMER"]

######### BRAND_NAME ANALYSIS ##################

groups = pd.get_dummies(final["group"])
brands = pd.get_dummies(final["BRAND_NAME"])
groups_brands = groups.join(brands)

groups_brands.shape

groups_brands.head()

###### Support-Confidence-Lift #####################

freq_groupsbrands = apriori(groups_brands, min_support=0.008, use_colnames=True)
rules = association_rules(freq_groupsbrands, metric="lift", min_threshold=0.5)
rules.sort_values("confidence", ascending=False, inplace=True)

set_temp = final["group"].unique()
rules[rules["antecedents"].apply(lambda x: list(x)).apply(lambda x: x in set_temp)]

rules[rules["antecedents"] == {"YOUNG SINGLES/COUPLES - Mainstream"}]


# From ASSOCIATION RULE,MARKET BASKET analysis, we can see that for Mainstream - young singles/couples,
# Kettle is the brand of choice. This is also true for most other segments. We can use the affinity index to see
# if there are brands this segment prefers more than the other segments to target.

# find the target rating proportion
target_segment = (
    young_mainstream["BRAND_NAME"]
    .value_counts()
    .sort_values(ascending=True)
    .rename_axis("BRANDS")
    .reset_index(name="target")
)
target_segment.target /= young_mainstream["PROD_QTY"].sum()

# find the other rating proportion
not_young_mainstream = df.loc[df["LIFESTAGE"] != "YOUNG SINGLES/COUPLES"]
not_young_mainstream = not_young_mainstream.loc[
    not_young_mainstream["PREMIUM_CUSTOMER"] != "Mainstream"
]
other = (
    not_young_mainstream["BRAND_NAME"]
    .value_counts()
    .sort_values(ascending=True)
    .rename_axis("BRANDS")
    .reset_index(name="other")
)
other.other /= not_young_mainstream["PROD_QTY"].sum()

# join the two dataframes
brand_proportions = target_segment.set_index("BRANDS").join(other.set_index("BRANDS"))
brand_proportions = brand_proportions.reset_index()
brand_proportions["affinity"] = brand_proportions["target"] / brand_proportions["other"]
brand_proportions.sort_values(by="affinity", ascending=False)

"""By using the affinity index, we can see that mainstream young singles/couples
are 24% more likely to purcahse Tyrrells chips than the other segments.
However, they are 49% less likely to purchase Burger Rings."""

# We also want to find out if our target segment tends to buy larger packs of chips.

# Plot the distribution of the packet sizes for a general indication of what it most popular.
young_mainstream = df.loc[df["LIFESTAGE"] == "YOUNG SINGLES/COUPLES"]
young_mainstream = young_mainstream.loc[
    young_mainstream["PREMIUM_CUSTOMER"] == "Mainstream"
]
ax = (
    young_mainstream["PACK_SIZE"].value_counts().sort_values(ascending=True).plot.barh()
)
ax.set_ylabel("Packet size (g)")
ax.set_xlabel("Packets purchased")
plt.show()


# Also want to check which brands correspond to what sized packets.
brand_size = young_mainstream.groupby(["BRAND_NAME", "PACK_SIZE"], as_index=False)[
    "TOT_SALES"
].agg(["sum"])
ax = brand_size.sort_values(by="sum").plot.barh(y="sum", figsize=(14, 14))
ax.set_ylabel("(Brand, packet size(g))")
ax.set_xlabel("Packets purchased")
plt.show()

############### PACK_SIZE ANALYSIS #########

groups = pd.get_dummies(final["group"])
brands = pd.get_dummies(final["PACK_SIZE"])
groups_brands = groups.join(brands)

groups_brands.shape
groups_brands.head()


freq_groupsbrands = apriori(groups_brands, min_support=0.009, use_colnames=True)
rules = association_rules(freq_groupsbrands, metric="lift", min_threshold=0.5)
rules.sort_values("confidence", ascending=False, inplace=True)
set_temp = final["group"].unique()
rules[rules["antecedents"].apply(lambda x: list(x)).apply(lambda x: x in set_temp)]

# While it appears that most segments purchase more chip packets that are 175g, which is also the size
# that most Kettles chips are purchased in, we can also determine whether mainstream young singles/couples have certain
# preferences over the other segments again using the affinity index.

# find the target rating proportion

target_segment = (
    young_mainstream["PACK_SIZE"]
    .value_counts()
    .sort_values(ascending=True)
    .rename_axis("SIZES")
    .reset_index(name="target")
)
target_segment.target /= young_mainstream["PROD_QTY"].sum()

# find the other rating proportion

other = (
    not_young_mainstream["PACK_SIZE"]
    .value_counts()
    .sort_values(ascending=True)
    .rename_axis("SIZES")
    .reset_index(name="other")
)
other.other /= not_young_mainstream["PROD_QTY"].sum()

# join the two dataframes

brand_proportions = target_segment.set_index("SIZES").join(other.set_index("SIZES"))
brand_proportions = brand_proportions.reset_index()
brand_proportions["affinity"] = brand_proportions["target"] / brand_proportions["other"]
brand_proportions.sort_values(by="affinity", ascending=False)

# Here, we can see that mainstream young singles/couples are 28% more likely to purcahse 270g chips than the
# other segments. However, they are 49% less likely to purchase 220g chips.
# The chips that come in 270g bags are Twisties while Burger Rings come in 220g bags,
# which is consistent with the affinity testing for the chip brands.


"""Summary of Insights
The three highest contributing segments to the total sales are:

Older families - Budget
Young singles/couples - Mainstream
Retirees - Mainstream
The largest population group is mainstream young singles/couples, followed by mainstream retirees which explains their large total sales. 
While population is not a driving factor for budget older families, older families and young families in general buy more chips per customer. 
Furthermore, mainstream young singles/couples have the highest spend per purchase, 
which is statistically significant compared to the non-mainstream young singles/couples. 
Taking a further look at the mainstream yong singles/couples segment, 
we have found that they are 24% more likely to purchase Tyrells chips than the other segments. 
This segment does purchase the most Kettles chips, which is also consistent with most other segments. 
However, they are 49% less likely to purchase Burger Rings, 
which was also evident in the preferences for packet sizes given they are the only chips that come in 220g sizes. 
Mainstream young singles/couples are 28% more likely to purchase 270g chips, which is the size that Twisties come in, 
compare to the other segments. The packet size purchased most over many segments is 175g.

Perhaps we can use the fact that Tyrells and (the packet size of) Twisties chips are more likely to be purchased 
by mainstream young singles/couples and place these products where they are more likely to be seen by this segment. 
Furthermore, given that Kettles chips are still the most popular, 
if the primary target segment are mainstream young singles/couples, 
Tyrells and Twisties could be placed closer to the Kettles chips. This strategy, 
with the brands they are more likely to purchase, could also be applied to other segments that purchase the most of Kettles to increase their total sales."""

## Импорты и считывание данных
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.3, rc={'figure.figsize':(14, 8)})
train_df = pd.read_csv("../data/train.csv", low_memory=False)
test_df = pd.read_csv("../data/test.csv", low_memory=False)
# ---
# Объединение двух таблиц(чтобы не дублировать код)
df = train_df.append(test_df, sort=False)
del train_df
del test_df

df = df.reset_index(drop=True)

test_mask = df.Sales.isna()
train_mask = ~test_mask
len(df)

## Работа с признаками

### Функции для добавления статистик
def add_feature_count(dataframe, feature, stat_mask=None, default=0):
    if stat_mask is None:
        counts = dataframe[feature].value_counts()
    else:
        counts = dataframe.loc[stat_mask, feature].value_counts()
        
    column_name = f"{feature} goods count"
    dataframe[column_name] = dataframe[feature].map(lambda x: counts.get(x, default))
    return dataframe
def add_feature_mean(dataframe, grouped_by, feature, stat_mask=None, default=0):
    if stat_mask is None:
        grouped_feature = dataframe.groupby(grouped_by)[feature]
    else:
        grouped_feature = dataframe[stat_mask].groupby(grouped_by)[feature]

    grouped_feature_mean = grouped_feature.mean()
    grouped_feature_std = grouped_feature.skew().fillna(default)
    
    column_name_mean = f"{grouped_by} mean {feature.lower()}"
    column_name_std = f"{grouped_by} std {feature.lower()}"
    
    dataframe[column_name_mean] = dataframe[grouped_by].map(lambda x: grouped_feature_mean.get(x, default))
    dataframe[column_name_std] = dataframe[grouped_by].map(lambda x: grouped_feature_std.get(x, default))
    
    return dataframe
QUANTILES = tuple(0.1 * i for i in range(11))

def add_feature_quantiles(dataframe, grouped_by, feature, stat_mask=None, default=0, quantiles=QUANTILES):
    if stat_mask is None:
        grouped_feature = dataframe.groupby(grouped_by)[feature]
    else:
        grouped_feature = dataframe[stat_mask].groupby(grouped_by)[feature]

    for quantile in QUANTILES:
        grouped_feature_quantile = grouped_feature.quantile(quantile)
        column_name = f"{grouped_by} {feature.lower()} quantile {quantile:.2f}"
        dataframe[column_name] = dataframe[grouped_by].map(lambda x: grouped_feature_quantile.get(x, default))

    return dataframe

# ---
# Проверим дубликаты
mask = df.duplicated()
(train_mask & mask).sum()
(test_mask & mask).sum()
# ---
# В обучающей выборке есть полные дубликаты, в тестовой их нет, скорее всего проблемы с обучающей выборкой  
# Удаление этих дубликатов
df = df.drop(df[mask].index)

df = df.reset_index(drop=True)

test_mask = df["Sales"].isna() 
train_mask = ~test_mask
# ---
# В обучающей выборке есть несколько товаров с "поломанными/Error" признаками - удалим их
df.loc[train_mask, "Deliveryscheme"].unique()
df.loc[test_mask, "Deliveryscheme"].unique()
mask = (df["Deliveryscheme"] == "0,0000") | (df["Deliveryscheme"] == "4,8000")
df[mask] 
df = df.drop(df[mask].index)

df = df.reset_index(drop=True)

test_mask = df["Sales"].isna() 
train_mask = ~test_mask
# Работа со столбцом Name, убираем лишние символы с помощью регулярнго выражения
df["Name"].head()
df["Name"] = df["Name"].str.lower()
df["Name"].head()
chars_to_delete = "[^a-zа-я0-9.,\-ёЁ]"
df["Name"] = df["Name"].str.replace(chars_to_delete, " ", regex=True)
df["Name"].head()

words_count = {}

def increase_word_count(name):
    global words_count
    
    for word in name.split():
        if len(word) > 3 and not word.isdigit():
            if word in words_count:
                words_count[word] += 1
            else:
                words_count[word] = 1

df["Name"].apply(increase_word_count);
import operator
sorted(words_count.items(), key=operator.itemgetter(1), reverse=True)[:100]
# ---
# Добавим признак - число штук в названии товара
df["Name num pieces"] = df["Name"].str.extract(r"(\d+)\s*шт(ук|\s|$|\.|,)")[0].fillna(0).astype(float)
threshold = df["Name num pieces"].mean() + 3 * df["Name num pieces"].std()
df.loc[df["Name num pieces"] > threshold, "Name num pieces"] = 0
df.loc[df["Name num pieces"] != 0, "Name num pieces"].describe()
# ---
# Добавим признак - число миллилитров в названии товара
df["Name num milliliters"] = df["Name"].str.extract(r"((\d+\.)?\d+)\s*мл([^a-zа-яё]|$)")[0].fillna(0).astype(float)
threshold = df["Name num milliliters"].mean() + 3 * df["Name num milliliters"].std()
df.loc[df["Name num milliliters"] > threshold, "Name num milliliters"] = 0
df.loc[df["Name num milliliters"] != 0, "Name num milliliters"].describe()
# ---
# Добавим признаки - число миллиметров, сантиметров в названии товара
df["Name num millimeters"] = df["Name"].str.extract(r"((\d+\.)?\d+)\s*мм([^a-zа-яё]|$)")[0].fillna(0).astype(float)
threshold = df["Name num millimeters"].mean() + 3 * df["Name num millimeters"].std()
df.loc[df["Name num millimeters"] > threshold, "Name num millimeters"] = 0
df.loc[df["Name num millimeters"] != 0, "Name num millimeters"].describe()
df["Name num centimeters"] = df["Name"].str.extract(r"((\d+\.)?\d+)\s*см([^a-zа-яё]|$)")[0].fillna(0).astype(float)
threshold = df["Name num centimeters"].mean() + 3 * df["Name num centimeters"].std()
df.loc[df["Name num centimeters"] > threshold, "Name num centimeters"] = 0
df.loc[df["Name num centimeters"] != 0, "Name num centimeters"].describe()
# ---
# Добавим признак - число килограмм в названии товара
df["Name num kilograms"] = df["Name"].str.extract(r"((\d+\.)?\d+)\s*кг([^a-zа-яё]|$)")[0].fillna(0).astype(float)
threshold = df["Name num kilograms"].mean() + 3 * df["Name num kilograms"].std()
df.loc[df["Name num kilograms"] > threshold, "Name num kilograms"] = 0
df.loc[df["Name num kilograms"] != 0, "Name num kilograms"].describe()
# ---
# Добавим признаки - в названии имеется подстрока $x$
def add_name_substr_feature(dataframe, substr):
    df[f"Name contains \"{substr}\""] = (
        df["Name"].str.contains(f"{substr}")
    ).astype(int)
    
    count = df[f"Name contains \"{substr}\""].sum()
    print(f"Подстроку в названии содержат {count} товаров")
    if count > 0:
        print("Несколько примеров: ")
        for example in df.loc[df[f"Name contains \"{substr}\""] == 1, "Name"].values[:5]:
            print(f"\t{example}")
add_name_substr_feature(df, "набор")
add_name_substr_feature(df, "комплект")
add_name_substr_feature(df, "игрушка")
add_name_substr_feature(df, "чехол")
add_name_substr_feature(df, "сумка")
add_name_substr_feature(df, "кроссовки")
add_name_substr_feature(df, "футболка")
add_name_substr_feature(df, "ботинки")
add_name_substr_feature(df, "игра")
add_name_substr_feature(df, "подушка")
add_name_substr_feature(df, "коврик")
add_name_substr_feature(df, "крем")
add_name_substr_feature(df, "органайзер")
add_name_substr_feature(df, "картина")
add_name_substr_feature(df, "аккумулятор")
add_name_substr_feature(df, "кабель")
add_name_substr_feature(df, "smart")
# ---
# Сделаем embedding'и названий при помощи word2vec'а
names = df["Name"]
names = names.str.replace("[^а-яё]", " ", regex=True)
names
names = names.apply(lambda x: " ".join(x.split()))
names
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim.models import Word2Vec
stemmer = SnowballStemmer("russian") 
russian_stopwords = stopwords.words("russian")

def preprocess_text(text):
    tokens = [word for word in text.split() if word not in russian_stopwords]
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens
%%time

names = names.apply(preprocess_text)
names
model = Word2Vec(sentences=list(names), vector_size=100, window=2, min_count=100, workers=4)
model.build_vocab(list(names), progress_per=10000)
model.train(list(names), total_examples=model.corpus_count, epochs=30, report_delay=1)
model.wv.most_similar("тетрад")
def make_embedding(text):
    res = np.zeros(100)
    cnt = 0
    for token in text:
        if token in model.wv:
            res += model.wv[token]
            cnt += 1
    if cnt != 0:
        return res / cnt
    else:
        return res
names = names.apply(make_embedding)
names = pd.DataFrame(names.tolist())
names = names.add_prefix("Name_")
df = df.join(names)
del names
### Category
# Некоторые товары имеют категорию верхнего уровня "Супермаркет Экспресс", являющуюся "метакатегорией", поскольку она означает быструю доставку
mask = df["Category"].str.contains("Супермаркет Экспресс")
df["Category supermarket express"] = mask.astype(int)
df.loc[mask, "Category"] = df.loc[mask, "Category"].str.replace("Супермаркет Экспресс(/|$)", "", regex=True)
df.loc[mask, "Category"]
### Brand
### Seller
df[df["Seller"] == "OZON, доставка OZON"] # проверка, т.к. после удаления ", доставка OZON" прибавился 1 товар продавца OZON 
df.loc[train_mask, "Seller"].value_counts()
df.loc[test_mask, "Seller"].value_counts()
# ---
# В наименованиях многих продавцов есть подстрока "доставка OZON"  
# Проверим, не равнозначно ли это какой-то схеме доставки
mask = df["Seller"].str.contains(", доставка OZON")
df.loc[mask, "Deliveryscheme"].value_counts()
# ---
# Не очень понятно, почему это не равносильно FBO (причем почти все товары с доставкой OZON имеют схему доставки FBS)  
# Возможно, это опция при покупке товара  
# Вынесем это в отдельный признак  
df["Ozon delivery"] = mask.astype(int)
# ---
# Удаление из наименования продавца эту подстроку  
df.loc[mask, "Seller"] = df.loc[mask, "Seller"].str.replace(", доставка OZON", "")
# ---
# Снова взглянем на число товаров продавца
df.loc[train_mask, "Seller"].value_counts()
df.loc[test_mask, "Seller"].value_counts()
# ---
# Добавим признак - (продавец - "ООО")
df["Seller"].str.contains("ООО").mean()
df["Seller contains OOO"] = df["Seller"].str.contains("ООО").astype(int)
# ---
# Добавим признак - (продавец - ИП)
df.loc[df["Seller"] == "ИП"]
mask = df["Seller"].str.contains("И\.П\.") | (df["Seller"] == "ИП") | \
        df["Seller"].str.startswith("ИП ") | df["Seller"].str.endswith(" ИП")
df.loc[mask, "Seller"]
df["Seller contains ip"] = mask.astype(int)
# Deliveryscheme
df["Deliveryscheme"].value_counts().plot.pie();
threshold = 100
sns.catplot(x="Deliveryscheme", y="Sales", kind="boxen", data=df[df["Sales"] <= threshold], height=8, aspect=14/8);
sns.catplot(x="Deliveryscheme", y="Sales", kind="boxen", data=df[df["Sales"] > threshold], height=8, aspect=14/8);
# ---
# Закодируем схему доставки при помощи OneHotEncoding
df = df.join(pd.get_dummies(df["Deliveryscheme"]))
df = df.drop("Deliveryscheme", axis=1)
df.head()
# Comments
# Проверим товары с огромным числом комментариев
df.loc[train_mask & (df["Comments"] > 10000)]
df.loc[test_mask & (df["Comments"] > 10000)]
# ---
# Наверное все нормально
# Rating
# Рейтинг записан через запятую, поэтому не воспринялся как числовой признак, исправим это
df["Rating"] = df["Rating"].str.replace(',', '.').astype(float)
df.describe()
# Price
# В обучающей выборке содержатся товары с текущей ценой 0 (в тестовой выборке таких нет)
mask = df["Price"] == 0
df[mask]
# ---
# Вряд ли это нормально, удалим их
df = df.drop(df[mask].index)
df = df.reset_index(drop=True)

test_mask = df.Sales.isna()
train_mask = ~test_mask
# ---
# Взглянем на товары с маленькой ценой
df.loc[train_mask & (df["Price"] < 20)]
df.loc[test_mask & (df["Price"] < 20)]
# ---
# Вроде все в порядке  
# Взглянем на товары с огромной ценой
df.loc[train_mask & (df["Price"] > 1000000)]
df.loc[test_mask & (df["Price"] > 1000000)]
# ---
# Ковры конечно дорогие, но не очень понятно что с ними делать (возможно так и должно быть), т.к. они есть и в тестовой выборке
# ---
# Добавим признак - нормированная текущая цена
df["Norm price"] = 0
df.loc[(df["Max price"] != df["Min price"]), "Norm price"] = (df["Price"] - df["Min price"]) / (df["Max price"] - df["Min price"])
### Max price
df.loc[train_mask & (df["Max price"] > 2000000)]
df.loc[test_mask & (df["Max price"] > 2000000)]
# ---
# Все аналогично `Price`
### Min Price
df.loc[train_mask & (df["Min price"] == 0)]
df.loc[train_mask & (df["Min price"] == 0), "Sales"].describe()
df.loc[test_mask & (df["Min price"] == 0)]
# ---
# Странно, что минимальная цена равна 0, но значения `Sales` в обучающей выборке для таких товаров различны  
# Также такие товары есть в тестовой выборке, оставим как есть
# Average Price
# Есть пропуски, взглянем на них
df.loc[df["Average price"].isna(), "Sales"]
# ---
# Если средняя цена NaN, то продаж не было
df[df["Sales"] == 0].describe()
# ---
# Если продаж не было, то средняя цена NaN
# Эти 2 условия равнозначны  

# Положим `Sales` для товаров в тестовой выборке с пропусками в `Average price` равным 0  
# Будем предсказывать `Sales` от 1
answer = pd.DataFrame({
    "Id": df[test_mask].Id, 
    "Expected": np.nan
})
answer.loc[df.loc[test_mask, "Average price"].isna(), "Expected"] = 0
answer.head()
answer.Id = answer.Id.astype(int)
answer = answer.set_index("Id")
answer.head()
answer["Expected"].count() / len(answer)
# ---
# ~ 9% ответов уже есть
mask = df["Average price"].isna()
df = df.drop(df[mask].index)

df = df.reset_index(drop=True)

test_mask = df["Sales"].isna() 
train_mask = ~test_mask
# ---
# Введем признаки - Нормированную среднюю цену и статистики по ней
df["Norm average price"] = 0
df.loc[(df["Max price"] != df["Min price"]), "Norm average price"] = (df["Average price"] - df["Min price"]) / (df["Max price"] - df["Min price"])
# Days in stock
# Из-за "поломанных" строк, удаленных ранее (в `Deliveryscheme`) тип не определился, поправим это
df["Days in stock"] = df["Days in stock"].astype(float)
# Sales
threshold = 50
plt.figure(figsize=(16, 10))
sns.histplot(data=df[df["Sales"] <= threshold], x="Sales", bins=10)
plt.show()
plt.figure(figsize=(16, 10))
sns.histplot(data=df[df["Sales"] > threshold], x="Sales", bins=100)
plt.show()
(df["Sales"] == 700).sum()
df[df["Sales"] == 700].describe()
df.describe()
df[(df["Sales"] == 700)]
df[(df["Sales"] == 700) & (df["Seller"] == "ООО \"Джой групп\"")]
df[test_mask & (df["Seller"] == "ООО \"Джой групп\"")]
# Сбор статистик
test_mask = df["Id"].notna()
train_mask = ~test_mask
train_val_indices = df[train_mask].index
np.random.seed(42)
np.random.shuffle(train_val_indices.to_numpy())
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(train_val_indices, train_size=0.9)
val_mask = np.zeros(df.shape[0], bool)
val_mask[val_indices] = True

train_mask = np.zeros(df.shape[0], bool)
train_mask[train_indices] = True
df = add_feature_mean(df, grouped_by="Brand", feature="Days in stock")
df = add_feature_quantiles(df, grouped_by="Brand", feature="Days in stock")

df = add_feature_mean(df, grouped_by="Seller", feature="Days in stock")
df = add_feature_quantiles(df, grouped_by="Seller", feature="Days in stock")
df = add_feature_mean(df, grouped_by="Brand", feature="Norm average price")
df = add_feature_quantiles(df, grouped_by="Brand", feature="Norm average price")

df = add_feature_mean(df, grouped_by="Seller", feature="Norm average price")
df = add_feature_quantiles(df, grouped_by="Seller", feature="Norm average price")
stat_mask = (df["Rating"] != 0)

df = add_feature_mean(df, grouped_by="Brand", feature="Rating", stat_mask=stat_mask)
df = add_feature_quantiles(df, grouped_by="Brand", feature="Rating", stat_mask=stat_mask)

df = add_feature_mean(df, grouped_by="Seller", feature="Rating", stat_mask=stat_mask)
df = add_feature_quantiles(df, grouped_by="Seller", feature="Rating", stat_mask=stat_mask)
df = add_feature_count(df, feature="Seller")

stat_mask = (df["Comments"] != 0)

df = add_feature_mean(df, grouped_by="Seller", feature="Comments", stat_mask=stat_mask)
df = add_feature_quantiles(df, grouped_by="Seller", feature="Comments", stat_mask=stat_mask)
df = add_feature_count(df, feature="Brand")

stat_mask = (df["Comments"] != 0)

df = add_feature_mean(df, grouped_by="Brand", feature="Comments", stat_mask=stat_mask)
df = add_feature_quantiles(df, grouped_by="Brand", feature="Comments", stat_mask=stat_mask)
df = add_feature_mean(df, grouped_by="Brand", feature="Sales", stat_mask=(train_mask | val_mask))
df = add_feature_quantiles(df, grouped_by="Brand", feature="Sales", stat_mask=(train_mask | val_mask))

df = add_feature_mean(df, grouped_by="Seller", feature="Sales", stat_mask=(train_mask | val_mask))
df = add_feature_quantiles(df, grouped_by="Seller", feature="Sales", stat_mask=(train_mask | val_mask))
# Обучение
# Проверка NaN
df.isna().sum()[df.isna().sum() != 0]
### Подготовка к обучению
target_column = "Sales"

feature_columns = list(df.dtypes[df.dtypes != "object"].index)
feature_columns.remove(target_column)
feature_columns.remove("Id")

feature_columns
X_train = df.loc[train_mask, feature_columns].values.astype("float32")
y_train = df.loc[train_mask, target_column].values.astype("float32")
X_val = df.loc[val_mask, feature_columns].values.astype("float32")
y_val = df.loc[val_mask, target_column].values.astype("float32")
X_test = df.loc[test_mask, feature_columns].values.astype("float32")
print(f"Товаров в обучающей выборке: {X_train.shape[0]}")
print(f"Товаров в валидационной выборке: {X_val.shape[0]}")
print(f"Товаров в тестовой выборке: {X_test.shape[0]}")
print(f"Число признаков, подаваемое модели: {X_train.shape[1]}")
# Метрика
# Используемая метрика - sMAPE (symmetric mean absolute percentage error):  
# $$sMAPE = \frac{100\%}{L}\sum_{i=1}^{L}{\frac{|\hat{y} - y|}{\frac{1}{2}(|\hat{y}| + |y|)}}$$  
# Подробнее: [wikipedia](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
def smape(y, y_pred):
    obj_scores = np.abs(y_pred - y) / (np.abs(y_pred) + np.abs(y))
    obj_scores = np.append(obj_scores, np.zeros(7600))
    return 200 * np.mean(obj_scores)
from sklearn.metrics import make_scorer

smape_scorer = make_scorer(smape, greater_is_better=False)
### Выбор модели
#### Среднее по обучающей выборке
smape(np.array([np.mean(y_train)] * len(y_val)), y_val)
#### Медиана по обучающей выборке
smape(np.array([np.median(y_train)] * len(y_val)), y_val)
#### Нейросеть (PyTorch)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import torch.nn.functional as F
import copy

DEVICE = torch.device("cuda:0")
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
dataset_train = CustomDataset(X_train, y_train)
dataset_val = CustomDataset(X_val, y_val)

loader_train = DataLoader(dataset_train, batch_size=10000)
loader_val = DataLoader(dataset_val, batch_size=10000)
def train_model(model, loader_train, loader_val, loss_fn, optimizer, epochs):
    best_model = None
    best_smape = None
    
    pbar = tqdm(range(epochs), desc='Train')
    for epoch in pbar:
        model.train()
        train_loss = 0
        train_smape = 0
        train_size = 0
        for x, y in loader_train:
            x_gpu = x.to(DEVICE)
            y_gpu = y.to(DEVICE)
            train_size += x_gpu.shape[0]
            pred = model(x_gpu)[:, 0]
            optimizer.zero_grad()
            loss = loss_fn(pred, y_gpu)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= train_size
        
        model.eval()
        val_loss = 0
        val_smape = 0
        val_size = 0
        with torch.no_grad():
            for x, y in loader_val:
                x_gpu = x.to(DEVICE)
                y_gpu = y.to(DEVICE)
                val_size += x_gpu.shape[0]
                pred = model(x_gpu)[:, 0]
                loss = loss_fn(pred, y_gpu)
                val_loss += loss.item()
            
        val_loss /= val_size
        
        if best_smape is None or val_loss < best_smape:
            best_smape = val_loss
            best_model = copy.deepcopy(model)
            
        pbar.set_description(f'({train_loss:.3f}, {val_loss:.3f})')
        
    return best_model, best_smape
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        sizes = (len(feature_columns), 180, 90, 1)
        self.bn = nn.ModuleList([nn.BatchNorm1d(x) for x in sizes[1:-1]])
        self.linear = nn.ModuleList([nn.Linear(x, y) for x, y in zip(sizes, sizes[1:])])
        
    def forward(self, x):
        x0 = F.relu(self.bn[0](self.linear[0](x))) # 180
        x1 = F.relu(self.bn[1](self.linear[1](x0))) # 90
        return self.linear[2](x1)
    
model = Model().to(DEVICE)
def smape_loss(y_pred, y):
    return 200 * torch.sum((y - y_pred).abs() / (y.abs() + y_pred.abs())) / 1.091

optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
model.apply(init_weights);
best_model, best_smape = train_model(model, loader_train, loader_val, smape_loss, optimizer, 200)
print(f"Best sMAPE: {best_smape:.4f} %")
(57 * 76000 + 200 * (83500 - 76000)) / 83500
## Конечный результат
y_test_pred = best_model.cpu()(torch.from_numpy(X_test))
y_test_pred = y_test_pred.detach().numpy()
(y_test_pred < 1).sum()
y_test_pred[y_test_pred < 1] = 1
df.loc[test_mask, "Sales"] = y_test_pred
df.loc[test_mask, ["Id", "Sales"]]
answer.loc[answer.Expected != 0, "Expected"] = df.loc[test_mask, "Sales"].values
answer
answer.to_csv("14-07-nn-score-54-5567-200-epochs.csv")
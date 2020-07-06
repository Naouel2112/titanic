{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users/balde/OneDrive/Documents/Cours M2/IPSSI/Cours Python/Projet du 01-07-2020/Projet Titanic/kaggle/input/titanic\\test.csv\n",
      "C:\\Users/balde/OneDrive/Documents/Cours M2/IPSSI/Cours Python/Projet du 01-07-2020/Projet Titanic/kaggle/input/titanic\\train.csv\n"
     ]
    }
   ],
   "source": [
    "### Import des bibliothèques \n",
    "import numpy as np\n",
    "import pandas as pd \n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.style.use('seaborn-notebook')\n",
    "\n",
    "# Normalisation et mise à l'échelle des données\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "import os\n",
    "\n",
    "### Chargement du dossier contenant les fichiers\n",
    "for dirname, _, filenames in os.walk('C:\\\\Users/balde/OneDrive/Documents/Cours M2/IPSSI/Cours Python/Projet du 01-07-2020/Projet Titanic/kaggle/input/titanic'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Lecture et affichage du train\n",
    "train_data = pd.read_csv(\"C:\\\\Users/balde/OneDrive/Documents/Cours M2/IPSSI/Cours Python/Projet du 01-07-2020/Projet Titanic/kaggle/input/titanic/train.csv\")\n",
    "train_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>male</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>330911</td>\n",
       "      <td>7.8292</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>female</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>363272</td>\n",
       "      <td>7.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>male</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>240276</td>\n",
       "      <td>9.6875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>315154</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>female</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3101298</td>\n",
       "      <td>12.2875</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name     Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    male   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)  female   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    male   \n",
       "3          895       3                              Wirz, Mr. Albert    male   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female   \n",
       "\n",
       "    Age  SibSp  Parch   Ticket     Fare Cabin Embarked  \n",
       "0  34.5      0      0   330911   7.8292   NaN        Q  \n",
       "1  47.0      1      0   363272   7.0000   NaN        S  \n",
       "2  62.0      0      0   240276   9.6875   NaN        Q  \n",
       "3  27.0      0      0   315154   8.6625   NaN        S  \n",
       "4  22.0      1      1  3101298  12.2875   NaN        S  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Lecture et affichage du test\n",
    "test_data = pd.read_csv(\"C:\\\\Users/balde/OneDrive/Documents/Cours M2/IPSSI/Cours Python/Projet du 01-07-2020/Projet Titanic/kaggle/input/titanic/test.csv\")\n",
    "test_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher les détails statistique du train\n",
    "train_data.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Total</th>\n",
       "      <th>%</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Cabin</th>\n",
       "      <td>687</td>\n",
       "      <td>77.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Age</th>\n",
       "      <td>177</td>\n",
       "      <td>19.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <td>2</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Fare</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Parch</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SibSp</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Pclass</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Survived</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>PassengerId</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Total     %\n",
       "Cabin          687  77.1\n",
       "Age            177  19.9\n",
       "Embarked         2   0.2\n",
       "Fare             0   0.0\n",
       "Ticket           0   0.0\n",
       "Parch            0   0.0\n",
       "SibSp            0   0.0\n",
       "Sex              0   0.0\n",
       "Name             0   0.0\n",
       "Pclass           0   0.0\n",
       "Survived         0   0.0\n",
       "PassengerId      0   0.0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## Afficher les données manquantes\n",
    "total = train_data.isnull().sum().sort_values(ascending=False)\n",
    "percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100\n",
    "percent_2 = (round(percent_1, 1)).sort_values(ascending=False)\n",
    "missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])\n",
    "missing_data.head(12)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Observation :\n",
    "## Embarqué n'a que 2 valeurs manquantes. \n",
    "## Cabine doit être étudiée plus en détail, car 77% d'entre elles sont manquantes.\n",
    "## Age a 177 valeurs manquantes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Première question qu'on s'est posé :\n",
    "### Combien de personnes ont embarqué à partir de différents ports (PClass) ? \n",
    "### Existe-t-il une corrélation entre le port d'embarquement et la survie ?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "S    0.722783\n",
       "C    0.188552\n",
       "Q    0.086420\n",
       "Name: Embarked, dtype: float64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Pourcentage d'embarquement sur les différents ports\n",
    "train_data['Embarked'].value_counts()/len(train_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "### Graphe d'embarquement\n",
    "sns.set(style=\"whitegrid\")\n",
    "sns.countplot( x='Embarked', data=train_data, hue=\"Embarked\", palette=\"Set2\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "### Graphe survie selon le port d'embarquement \n",
    "sns.set(style=\"whitegrid\")\n",
    "sns.countplot( x='Survived', data=train_data, hue=\"Embarked\", palette=\"Set2\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>445.357143</td>\n",
       "      <td>0.553571</td>\n",
       "      <td>1.886905</td>\n",
       "      <td>30.814769</td>\n",
       "      <td>0.386905</td>\n",
       "      <td>0.363095</td>\n",
       "      <td>59.954144</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Q</th>\n",
       "      <td>417.896104</td>\n",
       "      <td>0.389610</td>\n",
       "      <td>2.909091</td>\n",
       "      <td>28.089286</td>\n",
       "      <td>0.428571</td>\n",
       "      <td>0.168831</td>\n",
       "      <td>13.276030</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>S</th>\n",
       "      <td>449.527950</td>\n",
       "      <td>0.336957</td>\n",
       "      <td>2.350932</td>\n",
       "      <td>29.445397</td>\n",
       "      <td>0.571429</td>\n",
       "      <td>0.413043</td>\n",
       "      <td>27.079812</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          PassengerId  Survived    Pclass        Age     SibSp     Parch  \\\n",
       "Embarked                                                                   \n",
       "C          445.357143  0.553571  1.886905  30.814769  0.386905  0.363095   \n",
       "Q          417.896104  0.389610  2.909091  28.089286  0.428571  0.168831   \n",
       "S          449.527950  0.336957  2.350932  29.445397  0.571429  0.413043   \n",
       "\n",
       "               Fare  \n",
       "Embarked             \n",
       "C         59.954144  \n",
       "Q         13.276030  \n",
       "S         27.079812  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Regroupement par moyenne et par port d'embarquement\n",
    "train_data.groupby('Embarked').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Deuxième question qu'on s'est posé :\n",
    "\n",
    "### Le sexe a t'il une influence sur la survie du passager ?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>female</th>\n",
       "      <td>431.028662</td>\n",
       "      <td>0.742038</td>\n",
       "      <td>2.159236</td>\n",
       "      <td>27.915709</td>\n",
       "      <td>0.694268</td>\n",
       "      <td>0.649682</td>\n",
       "      <td>44.479818</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>male</th>\n",
       "      <td>454.147314</td>\n",
       "      <td>0.188908</td>\n",
       "      <td>2.389948</td>\n",
       "      <td>30.726645</td>\n",
       "      <td>0.429809</td>\n",
       "      <td>0.235702</td>\n",
       "      <td>25.523893</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        PassengerId  Survived    Pclass        Age     SibSp     Parch  \\\n",
       "Sex                                                                      \n",
       "female   431.028662  0.742038  2.159236  27.915709  0.694268  0.649682   \n",
       "male     454.147314  0.188908  2.389948  30.726645  0.429809  0.235702   \n",
       "\n",
       "             Fare  \n",
       "Sex                \n",
       "female  44.479818  \n",
       "male    25.523893  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Regroupement par moyenne et par sexe\n",
    "train_data.groupby('Sex').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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5GJpArWIiIgENIUhERERCWgKQyIiIhLQFIZEREQkoCkMiYiISEBTGBIREZGApjAkIiIiAU1hSERERAKawpCIiIgENIUhERERCWgKQyIiIhLQGsWu9SKBIjExkYSEBDyeqr/HvPLKK3To0KFa11i+fDlPP/00c+bMqXEtS5cupVWrVhf1/rS0NKZPn84777xTozqWL1/Oq6++yqFDhwgJCaF169bcf//9DBw4sEbXFZHAoTAk0sC8/fbbFx1AGpsvvviCX//617z00kv07dsXgDVr1jBlyhSeeOIJrrnmGpcrFJGGQGFIpJFYvnw5f/jDH2jbti07d+4kIiKCH//4x7zzzjvs3LmTkSNH8uijjwJQVFTE5MmT2b17N82bN+epp56iS5cu7Ny5k6eeeorCwkJyc3Pp3r07f/zjHwkLC6N3796MGDGCzZs388ILL1TeNzc3l3vuuYfbb7+dO++8k+3btzNt2jSOHTuG1+vlrrvuYty4cQC8/PLLzJ49m6ioKDp37nzGr2PJkiX87ne/O+38L3/5S6666qoq555//nkeeeSRyiAE0LdvXx599FF+//vfKwyJSPVYEWkwEhIS7OjRo+1NN91U+d/9999vrbV22bJltkePHnbDhg3WWmt/+MMf2vHjx9vS0lJ75MgR26tXL3vo0CG7bNky2717d5uRkWGttfYf//iHHTdunLXW2ueee87OnDnTWmttWVmZHT16tE1LS6u898cff1yllo0bN9pRo0bZWbNmWWutLS8vt6NGjbKZmZnWWmvz8/PtDTfcYFevXm0//fRTO2rUKHvixAlbXl5uf/zjH9vvfe97F/3/4tixYzYhIcHm5eWd1lZQUGATEhLssWPHLvr6IhI4NDIk0sCc62OyDh060LNnTwA6depEs2bNCA0NpVWrVjRp0oTjx48Dznyf/v37A3DzzTfz61//mhMnTvDggw+yePFiXn/9dXbt2kVOTg5FRUWV1x8wYECV+02aNIm4uDjGjBkDwK5du9izZ0/lCBRASUkJGzduZPv27Vx33XU0bdoUgLFjx55xvtCFjAydj8/nu6D+IhKYFIZEGpHQ0NAqx8HBZ/4n/u0J2MYYgoOD+cUvfoHX6+WGG25g2LBhHDx4EGttZb/IyMgq73vqqad49dVXefPNN/nBD36A1+ulWbNmzJo1q7LP4cOHadasGc8//3yV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sFKDZejo0aMaNWqUJOnzzz/XkCFDFBISovbt26u0tDQgAQEAAKzUYBkKCfnh8tatW9WvXz/fscfjsS4VAABAgDRYhmJjY1VQUKBNmzapuLjYV4a2bNmihISEi3746tWrNXz4cKWnp+uNN9644Ps+/vhjDR48+BKjAwAANF2Dq8l++ctfatKkSSotLdWvf/1rOZ1OLV68WK+88opefvnlBj+4sLBQ8+fP1/LlyxUREaHx48drwIAB6tq1a533HTt2TM8//3zTfxMAAIBGaHBkqHfv3lq/fr1yc3M1adIkSTVPsH/77bfVv3//Bj84NzdXAwcOVFxcnJxOp4YOHars7Ozz3jdjxgw9/vjjjf8NAAAAmuCi+wxFREQoIiLCd9y3b1+/PrioqEjx8fG+Y5fLpe3bt9d5z5IlS9SjRw/16tXL37wNys/PvyyfAwBAc5KamhrsCM2aXztQN4bX65VhGL5j0zTrHO/Zs0c5OTn629/+pu++++6yfCfPJgMAAJeqUc8m80diYqKKi4t9x8XFxXK5XL7j7OxsFRcX6yc/+YkefvhhFRUV6f7777cqDgAAQL0sK0NpaWnKy8tTSUmJysvLlZOTo0GDBvmuZ2Zmau3atVq5cqUWLlwol8ulpUuXWhUHAACgXpaVoYSEBE2dOlUTJ07UPffco4yMDKWkpGjKlCnasWOHVV8LAABwSQzTNM1gh2gqj8ej/Px85gwBAIBLZtnIEAAAQHNAGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZGGQIAALZmaRlavXq1hg8frvT0dL3xxhvnXf/ggw909913a9SoUXr00Ud18uRJK+MAAACcx7IyVFhYqPnz52vp0qV655139NZbb2nfvn2+66Wlpfrd736nhQsXatWqVUpKStKCBQusigMAAFAvy8pQ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\n",
      "text/plain": [
       "<Figure size 591.25x972 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "### Graphe survie selon le sexe et le port d'embarquement \n",
    "FacetGrid = sns.FacetGrid(train_data, row='Embarked', height=4.5, aspect=1.6)\n",
    "FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None, palette=\"Set2\" )\n",
    "FacetGrid.add_legend();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA6QAAAH0CAYAAAAqruADAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nOzde5SdZX0v8G+SmQkhNxJIQsJlFIwg5Ag0VEJVKFoyQhKClnO4KPSUY9F1FNTTajFSUSoUKeuwjqLtsgd7qtLKRe4CkSPFU4hESBUaoRKFDIRECEnIkEky133+SDPkMvfZM+/syeezVtaavZ/3fd7fs/e7929/Z97MjCqVSqUAAADAEBtddAEAAADsmwRSAAAACiGQAgAAUAiBFAAAgEIIpAAAABRCIAUAAKAQAikMoqOOOiqLFi3K4sWLO/594QtfGPTjXn755bnpppsG/TgAMJKsWbMmRx11VD7ykY/sNXb55ZfnqKOOysaNG7vcX/+FvqsqugAY6f7hH/4hU6dOLboMAKAXxo4dmxdeeCEvv/xyDjnkkCTJ1q1b86//+q8FVwYjk0AKBfnNb36Tq6++Oq+//nra2tpy4YUX5pxzzsny5cvzP//n/8zMmTPzwgsvZNy4cbnkkkvy3e9+Ny+88ELmz5+fJUuWpL29Pddcc02eeuqpNDY2plQq5Stf+Urmzp3bq+MAAHsbM2ZMzjjjjNx77735+Mc/niT50Y9+lPe///359re/3dFv9V8oD4EUBtkf/dEfZfToN6+O//a3v53Jkyfnsssuy3XXXZdjjz02b7zxRs4999y87W1vS5L827/9W6688socc8wx+ehHP5pvfetb+c53vpMtW7bklFNOyX/7b/8ta9euzauvvppbbrklo0ePzre+9a383d/93W4NsbW1tcvjHH/88UP+WABAJTj77LPz2c9+tiOQ3nXXXVmyZEm+/e1v54UXXtB/oYwEUhhknV2y++tf/zovvvhilixZ0nHf9u3b88wzz+TII4/MoYcemmOOOSZJcvjhh2fixImpqanJ1KlTM378+GzevDknnHBCJk+enO9///t56aWXsnz58owfP36346xevbrL42iIANC5OXPmZMyYMVm5cmUOPPDANDY25u1vf3uS5IgjjsinP/1p/RfKRCCFArS1tWXixIm5++67O+577bXXMnHixPziF79ITU3NbttXVe39Un3kkUdy9dVX54//+I/z/ve/P0cccUTuueeeXh8HAOjaWWedlXvuuSdTp07N4sWLO+7/yU9+km9+85v6L5SJ37ILBXjrW9+a/fbbr6NRrVu3LgsXLszKlSt7Pcdjjz2W0047LRdccEHmzJmT//t//2/a2trKfhwA2BctXrw4Dz74YO6///4sXLiw4/5/+7d/03+hjARSKEBNTU2++c1v5vbbb8+iRYty8cUX51Of+tRevxChO+edd15+9rOfZdGiRfngBz+Yww47LGvWrEl7e3tZjwMA+6IZM2bkyCOPzFve8pYccMABHfefeeaZ+i+U0ahSqVQquggAAAD2PX5CCgAAQCEEUgAAAAohkAIAAFAIgRQAAIBCFPp3SNvb29PY2Jjq6uqMGjWqyFIAGAFKpVJaWloyfvz4jB7te679oTcDUE499eZCA2ljY2Oee+65IksAYAR6+9vf7g/Q95PeDMBg6Ko3FxpIq6urk+worqampk/7rly5MnPmzBmMsoalfWm91jpy7UvrtdZiNDc357nnnuvoL/TdQHrzrobTedEXlVp3Urm1q3voVWrt6h565ai9p95caCDdeSlQTU1Nxo4d2+f9+7NPJduX1mutI9e+tF5rLY5LTftvoL15V8PtvOitSq07qdza1T30KrV2dQ+9ctXeVW/2H2wAAAAohEAKAABAIQq9ZBegErW3t2fNmjVpbGzscpuqqqo8++yzQ1hVcYpa6/jx43PooYf6bboA9Ko3F6WSPxP0tfb+9GaBFKCPXnvttYwaNSpHHXVUl2+4jY2NGT9+/BBXVowi1tre3p6XX345r732WqZPnz6kxwZg+OlNby5KJX8m6Evt/e3Nw+vZAqgAr7/+embMmDHsGt6+ZPTo0ZkxY0Y2b95cdCkADAN6c/H625s9YwB91NbW5s+KDAPV1dVpbW0tugwAhgG9eXjoT28WSAH6wZ8VKZ7nAIBd6QvF689z4P+QAgzQpk1JQ8Pu9zU3V6emZmDzTpqUTJkysDn66p/+6Z+SJOeff/6A5rnwwgvzyU9+MieddFI5ygKAPumsN5eD3lx+AinAADU0JEuX7n5fU1MpA/070nV1Q9/0BtrsAGA46Kw3l4PeXH4CKUAF++1vf5s/+7M/y9atWzN69OhcccUV+R//43/kO9/5Tg499NAsX748N954Y7773e/mwgsvzOTJk7Nq1aosWrQomzZtyl/8xV8kSa699tocfPDBeeONN5IkkydPTn19/V7j//k//+dcddVVWbVqVdra2vInf/InOe2009Lc3JwvfOELWblyZQ455JBs2rSpsMcEAIq0Z2/+0z/903zhC18Y0t68cOHCiunN/g8pQAW7/fbb8/u///u54447ctlll2XFihXdbn/UUUdl6dKlueCCC/LQQw+lra0tpVIpP/rRj7JgwYKO7RYuXNjp+N/8zd/k2GOPzR133JGbb745f/u3f5s1a9bku9/9bpLkgQceyBVXXJEXX3xxUNcNAMPVnr35F7/4RbfbD0ZvfumllyqmN/sJKUAFO/nkk3PppZfm2WefzamnnpqPfOQjufnmm7vc/p3vfGeSZOrUqTn66KOzfPnyVFdX561vfWumTZvWsV1X48uWLcv27dvzgx/8IEmydevW/OY3v8nPfvaznHvuuUmSt7zlLTnhhBMGcdUAMHzt2ZvPPffc3HbbbV1uPxi9edWqVRXTmwVSgAo2d+7c/PCHP8wjjzyS+++/P3feeWeSpFQqJclev3p9v/326/h68eLFuf/++1NdXZ1FixbtNXdn4+3t7fnrv/7rHHvssUl2/CHyqqqq3HPPPR3HTJKqKu0FgH3Tnr359ttvTzK0vXny5Mm59dZbK6I3u2QXoIJdd911ueeee/LBD34wX/ziF/PMM89kypQp+fWvf50k+fGPf9zlvu9///vzxBNP5LHHHsvpp5/eq/F58+Z1/La/V199NWeddVZ++9vf5uSTT869996b9vb2vPzyy/nXf/3XQVgtAAx/e/bmf//3fx/y3rxu3bqK6c3DMyYD0CsXXnhh/vRP/zR33HFHxowZk69+9asZNWpU/vIv/zI33nhj3vOe93S573777Zff+Z3fSXNzc8aPH9+r8U9+8pP50pe+lIULF6atrS2f/exnc9hhh+XII4/MqlWrcsYZZ+SQQw7J29/+9kFbMwAMZ3v25quuuirjxo0b0t58+OGH54ILLqiI3iyQAgzQpEk7fg38rpqbR5Xl75D2ZObMmfnHf/zHve4/9dRT97pv5y832NVf/dVf7Xb70ksv7XZ8woQJuf7663e7r7GxMdXV1fnKV77Sc8EAMAQ6683lmrcne/bmxsbGjB8/fkh7c5KK6c0CKcAATZmy998ka2xsyfjxA0ykAEC/dNabGZ78H1IAAAAK4SekVKRNm5KGhvLNN2mS76IBMNxsStJds5uURPMCKptASkVqaEiWLi3ffHV1AikAw01Dku6aXV0EUqDSuWQXAACAQgikAAAAFEIgBQAAoBACKcCAbUpSv9u/6uq1e93X93+bBqXaz3/+83n55ZcHZe5dvfLKK/mTP/mTAc9zxx135PLLLy9DRQDsO/buzeX5pzcn5e3NfqkRwIDt/YtHSqWmJGMHOO/g/MKS5cuX5xOf+ETZ593TjBkz8nd/93eDfhwA2FtPvxSsv/TmcvMTUoAKtnz58lx88cX57//9v6euri6XXXZZmpubkyQ/+MEPsnDhwixatCiXX355Ghsb861vfSuvvvpqLrnkkmzatPt3eb/61a/mrLPOytlnn50bb7wxSfL1r389X//61zu2ed/73pc1a9bkjjvuyIUXXphFixblmmuuybvf/e60tLQkSZ577rmcddZZWbNmTd73vvdl06ZNnY4nyV133ZUPfvCDWbx4cZYsWZKmpqaO++vq6vKHf/iHeeSRRwb1MQSActqzN3/2s58d8t585ZVXlqU3f/nLXx703iyQAlS4n//85/niF7+YBx54IGvXrs2jjz6aX/3qV/nbv/3bfPe73829996bcePG5cYbb8wll1yS6dOn51vf+lam7PK3jl5++eX8v//3/3LPPffkn/7pn/LrX/+6owF15ZVXXsmdd96ZJUuW5J3vfGceffTRJMkPf/jDjqaWJFOmTOl0fNWqVbn11lvz/e9/P3fffXcOPPDA3HTTTXnllVdy/fXX5+abb84tt9ySxsbGQXjUAGDw7Nqbf/vb3w55b/7yl79clt48derUQe/NAilAhZs9e3YOPvjgjB49OkceeWQ2b96cJ554IqeddlpHYzv33HPz+OOPdznHjBkzMnbs2Jx33nn5zne+kz/7sz/L2LHdX3J8zDHHpKpqx//8OOuss/LDH/4wSfLAAw9k0aJFu23b2fjy5ctTX1+f//Jf/ksWL16cH//4x3n++efz85//PCeccEIOOuigVFVV7TUXAAx3u/bmt771rRXbmx955JFB780CKUCF27U5jRo1KqVSKe3t7bttUyqV0tra2uUcVVVVue222/KpT30qr7/+es4777y88MILHfPttPPSniTZb7/9Or5+//vfnyeeeCJPPPFEZs6cmRkzZuw2f2fjbW1tOeOMM3L33Xfn7rvvzm233ZYvfvGLex1zZ2MFgEoxUnrzd7/73UHvzQIpwAj0rne9Kw8//HBef/31JMmtt96ak046KUkyZsyYtLW17bb9M888k4985CP53d/93fz5n/95jjzyyLzwwguZMmVKfv3rXydJnn766axfv77T49XU1OS9731vrrnmmt0uCepu/KSTTspDDz2UDRs2pFQq5Utf+lL+4R/+IXPnzs0vfvGLvPLKK2lvb8/9999ftscFAIpSib35mmuuGfTe7NvOAAM2KTt+696bRo1qTlJThnn75+ijj87HPvaxXHjhhWlpacmxxx6bL3/5y0mS3//9388ll1yS//2//3cOO+ywJDsu8Tn++OOzcOHCjBs3Lr/zO7+TU045JW+88UaWLl2aM888M8cee2yOOeaYLo+5ePHi3HPPPamrq+vV+NFHH51PfvKT+aM/+qO0t7fnHe94Ry655JKMHTs2V1xxRf7rf/2vGTduXN72trf1+3EAYF+1d28u37z9U4m9efbs2YPem0eVdv3Z6xBramrKypUrM2fOnB6vh97TihUrMnfu3EGqbPjZl9bbm7XW1ydLy/ibvOvqktra8s3XW/vS85qMnPU+++yzecc73tHtNo2NjRk/fvwQVVSsIte653MxkL7CDuV6DCv19T686q5P93+2oi7Jm81reNXee+oeepVae3d196Y3F6WSPxP0p/a+9mY/IQUAqEit2RFad5g1q2W32ztMymD8zUSAchFIAQAqUmOSN39DZ3NzfXb9iekOdRFIgeHMLzUCAACgEAIpQD8U+N/v+Q+eAwB2pS8Urz/PgUAK0Ef77bdfx69DpxilUikbNmzY7e+tAbDv0puL19/e7P+QAvTRoYcemjVr1nT5d7+SpLm5OTU1A/2zL5WhqLXut99+OfTQQ4f8uAAMP73pzUWp5M8Efa29P71ZIAXoo+rq6rz1rW/tdpsVK1bkuOOOG6KKirUvrRWA4ak3vbkoldwnh6J2l+wCAABQCIEUAACAQgikAAAAFEIgBQAAoBACKQAAAIUQSAEAACiEQAoAAEAhBFIAAAAKIZACAABQCIEUAACAQgikAAAAFEIgBQAAoBACKQAAAIUQSAEAACiEQAoAAEAhBFIAAAAKIZACAABQCIEUAACAQgikAAAAFKKqNxvdeOONeeCB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m+5C7b6nMUnXnz+Srnt3Q0OytPuP+n32zndWl3fCTrhkFwAAgEIIpAAAABRCIAUAAKAQAikAAACFEEgBAAAohEAKAABAIQRSAAAACiGQAgAAUAiBFAAAgEIMKJDee++9OfPMMzN//vzcfPPN5aoJAOgnvRmASlLV3x1feeWV3HDDDbnjjjtSU1OT8847LyeddFLe9ra3lbM+AKCX9GYAKk2/f0K6bNmyzJs3LwcccED233//1NXV5cEHHyxnbQBAH+jNAFSafv+E9NVXX820adM6bk+fPj1PP/10n+YolUpJkubm5n7V0NTUtNvttrakpqarY7Vl1KguBv9jvKamaY/7dp+vP3PsOd7WVpNRozofb20tdTn/mDH7dTre1paO+Xpz/O7q31lDe3v/auxqvK81jhmzX5fH2Dn/rnOWu8b+7N+XGnfdv7O1djV/Xx7H3trzHH/z/u7PlZ5q7Gp853p7egx3PUZvHsdy1tif+TurccyY/fr9PHX1vPR9nrZuaxzIebRrjfvvX57zsa0taRrgNDv7yc7+sq8ZDr15V01NTbv15v6+t7xZW1tKpaZuXx/9Ocaur4+u+u2ex+i+lw18rXtu09lreM85+tNT+lpHd+P9fb/uzWecctXZ3Wec3s6xaw/v7HPdQPtLW9uO12G5zp+uetRAzvOdPaBc53lvP3t29Rmxs+O0tOzaU0pJemqsPW3TlqSnJtXW6Rxtbfunqalm0OvoLgv1X+temauveurNo0r97Np/8zd/k6ampnz6059Oktx6661ZuXJlrrrqql7P8cYbb+S5557rz+EBoEtvf/vbM3HixKLLGHJ6MwDDVVe9ud8/IT344IPz5JNPdtxev359pk+f3qc5xo8fn7e//e2prq7OqO6+XQUAvVAqldLS0pLx48cXXUoh9GYAhpueenO/A+nv/d7v5etf/3o2btyYcePG5Uc/+lH+8i//sk9zjB49ep/8DjYAg2e//fYruoTC6M0ADEfd9eZ+B9IZM2bkM5/5TC666KK0tLTknHPOyTvf+c7+TgcADJDeDECl6ff/IQUAAICB6PeffQEAAICBEEgBAAAohEAKAABAIQRSAAAACiGQAgAAUAiBFAAAgEIIpAAAABRCIAUAAKAQFRlI77333px55pmZP39+br755qLLGRRbtmzJwoULs2bNmiTJsmXLsmjRosyfPz833HBDwdWVz4033pgFCxZkwYIFue6665KM3LUmyf/6X/8rZ555ZhYsWJC///u/TzKy15skX/3qV3P55ZcnSZ599tl86EMfSl1dXb7whS+ktbW14OrK48ILL8yCBQuyePHiLF68OE899dSIfZ96+OGH86EPfShnnHFGvvKVryQZ+ecwfVNp536l9ttK7Z+V3gcrsadVao+q1H5z2223dTzWixcvzty5c3PVVVdVRO133313x/vKV7/61SRDdJ6XKsxvf/vb0mmnnVbatGlTqbGxsbRo0aLSqlWrii6rrH7xi1+UFi5cWDr22GNLL730Umnbtm2lU089tfTiiy+WWlpaShdffHHpkUceKbrMAXvsscdK5557bqmpqanU3Nxcuuiii0r33nvviFxrqVQqLV++vHTeeeeVWlpaStu2bSuddtpppWeffXbErrdUKpWWLVtWOumkk0p//ud/XiqVSqUFCxaUfv7zn5dKpVLp85//fOnmm28usryyaG9vL73nPe8ptbS0dNw3Ut+nXnzxxdJ73vOe0rp160rNzc2l888/v/TII4+M6HOYvqm0c79S+22l9s9K74OV2NMqtUeNlH7z3HPPlU4//fTS2rVrh33tW7duLf3u7/5uacOGDaWWlpbSOeecU3rssceG5DyvuJ+QLlu2LPPmzcsBBxyQ/fffP3V1dXnwwQeLLqusbr311lx55ZWZPn16kuTpp59ObW1tDjvssFRVVWXRokUjYs3Tpk3L5ZdfnpqamlRXV+fII4/M6tWrR+Rak+Rd73pXvvOd76SqqiobNmxIW1tbGhoaRux6X3/99dxwww35+Mc/niR5+eWXs3379hx//PFJkg996EMjYq3PP/98kuTiiy/OWWedle9973sj9n3qoYceyplnnpmDDz441dXVueGGGzJu3LgRew7Td5V27ldqv63U/lnJfbBSe1ql9qiR0m++9KUv5TOf+UxeeumlYV97W1tb2tvbs23btrS2tqa1tTVVVVVDcp5XXCB99dVXM23atI7b06dPzyuvvFJgReV39dVX58QTT+y4PVLXPHv27I4TfPXq1XnggQcyatSoEbnWnaqrq/O1r30tCxYsyMknnzxin9sk+eIXv5jPfOYzmTRpUpK9z+Np06aNiLU2NDTk5JNPzje+8Y38n//zf/L9738/a9euHZHPa319fdra2vLxj388ixcvzj/+4z+O6HOYvqu086FS+20l989K7YOV2tMqtUeNhH6zbNmybN++PWeccUZF1D5hwoR86lOfyhlnnJFTTz01hxxySKqrq4fkPK+4QNre3p5Ro0Z13C6VSrvdHolG+ppXrVqViy++OJ/73Ody2GGHjei1Jslll12Wn/70p1m3bl1Wr149Itd72223ZebMmTn55JM77hup5/EJJ5yQ6667LhMnTszUqVNzzjnn5Gtf+9qIXGtbW1t++tOf5pprrsktt9ySp59+Oi+99NKIXCv9U+mv80qrv1L7Z6X1wUruaZXao0ZCv/n+97+fP/7jP05SGefLv//7v+cHP/hB/vmf/zn/8i//ktGjR+exxx4bkrqryj7jIDv44IPz5JNPdtxev359x6U2I9XBBx+c9evXd9weSWtesWJFLrvssixZsiQLFizIz372sxG71t/85jdpbm7OO97xjowbNy7z58/Pgw8+mDFjxnRsM1LWe//992f9+vVZvHhxNm/enK1bt2bUqFG7PbevvfbaiFjrk08+mZaWlo4PKqVSKYcccsiIPI8POuignHzyyZk6dWqS5A/+4A9G7DlM/1R6j66kfluJ/bNS+2Al97RK7VGV3m+am5vzxBNP5Nprr01SGe8tjz76aE4++eQceOCBSXZcnnvTTTcNyXlecT8h/b3f+7389Kc/zcaNG7Nt27b86Ec/yimnnFJ0WYPquOOOywsvvNBx+cJ99903Ita8bt26fOITn8j111+fBQsWJBm5a02SNWvW5Iorrkhzc3Oam5vz4x//OOedd96IXO/f//3f57777svdd9+dyy67LO973/vyV3/1Vxk7dmxWrFiRZMdvchsJa33jjTdy3XXXpampKVu2bMmdd96Zv/7rvx6R71OnnXZaHn300TQ0NKStrS3/8i//kg984AMj8hymfyq9R1dKD6rU/lmpfbCSe1ql9qhK7ze/+tWv8pa3vCX7779/ksp4fR599NFZtmxZtm7dmlKplIcffjjvete7huQ8r7ifkM6YMSOf+cxnctFFF6WlpSXnnHNO3vnOdxZd1qAaO3Zsrr322lx66aVpamrKqaeemg984ANFlzVgN910U5qamjq+e5Qk55133ohca5Kceuqpefrpp3P22WdnzJgxmT9/fhYsWJCpU6eOyPV25vrrr88VV1yRLVu25Nhjj81FF11UdEkDdtppp+Wpp57K2Wefnfb29lxwwQWZO3fuiHyfOu644/LRj340F1xwQVpaWvLud787559/fo444oh95hyme5Xeoyul31Zq/xxpfbASelql9qhK7zcvvfRSDj744I7blfDe8p73vCfPPPNMPvShD6W6ujr/6T/9p1xyySU5/fTTB/08H1UqlUplnxUAAAB6UHGX7AIAADAyCKQAAAAUQiAFAACgEAIpAAAAhRBI+f/t3btKYwEUheFfMgiiBIOgiD6AaGsZEK1CbNRCC4kgWPkEErTRNoh1Gi2UgIRAtPMShIAIFkG08/ICFha5gFHjFAMy0088If5fdZoDe1eLda6SJEmSFAgLqdRC3t7eiEajrKysBD2KJEnCbJaazUIqtZDT01NGRka4u7vj8fEx6HEkSfrxzGapufwPqdRCEokE8Xic+/t73t/f2dzcBCCdTpPNZunu7mZ8fJzz83MKhQL1ep1UKsX19TUfHx+Mjo6yvr5OT09PwJtIktQezGapubxDKrWIh4cHSqUSsViMmZkZ8vk8Ly8vFItFcrkc2WyWXC5HtVr9OiedThMKhcjlchwdHdHf308qlQpwC0mS2ofZLDXfr6AHkPRHJpNhcnKSSCRCJBJheHiYw8NDnp+ficVihMNhABYXF7m6ugLg4uKCcrnM5eUl8Oc9l76+vsB2kCSpnZjNUvNZSKUWUKvVyOfzdHZ2MjU1BUClUmF/f5/p6Wn+frI+FAp9HTcaDZLJJBMTEwBUq1VeX1+/d3hJktqQ2Sx9Dx/ZlVrA8fExvb29FItFCoUChUKBs7MzarUaY2NjnJycUC6XAchms1/nRaNRDg4OqNfrNBoNNjY22N7eDmoNSZLahtksfQ8LqdQCMpkMy8vL/1xhDYfDJBIJ9vb2mJ+fZ2Fhgbm5OcrlMl1dXQCsrq4yNDTE7Ows8Xicz89P1tbWglpDkqS2YTZL38Ov7Eot7vb2llKpxNLSEgC7u7vc3Nyws7MT8GSSJP1MZrP0/1hIpRZXqVRIJpM8PT3R0dHB4OAgW1tbDAwMBD2aJEk/ktks/T8WUkmSJElSIHyHVJIkSZIUCAupJEmSJCkQFlJJkiRJUiAspJIkSZKkQFhIJUmSJEmBsJBKkiRJkgLxG116iE+FqkdVAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1152x576 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "##### Graphe survie selon l'age et le sexe\n",
    "survived = 'survived'\n",
    "not_survived = 'not survived'\n",
    "fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))\n",
    "women = train_data[train_data['Sex']=='female']\n",
    "men = train_data[train_data['Sex']=='male']\n",
    "ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False, color=\"blue\")\n",
    "ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False, color=\"yellow\")\n",
    "ax.legend()\n",
    "ax.set_title('Female')\n",
    "ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False, color=\"blue\")\n",
    "ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False, color=\"yellow\")\n",
    "ax.legend()\n",
    "_ = ax.set_title('Male');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "### Graphe survie selon la classe\n",
    "sns.barplot(x='Pclass', y='Survived', data=train_data, palette=\"Set2\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "### On observe que la classe a une forte influence sur la survie\n",
    "### surtout si cette personne est en classe 1. Nous allons créer un autre graphique pclass ci-dessous."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAuUAAAKnCAYAAAAhqjOZAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nOzdf1TUdaL/8dfEAKnl7wRKo2PuRkppst9Sj8lZKjAENRV/RDft7qbdSrY2K3XZbHNbTSzWLNMtzat1SuXgj/V6yY6m3cIfG5aGt3JNwbUUJy1ZUoZfn+8f3mZFUWaGGd4fh+fjHM9hhvfnM6/5jLx58eE9HxyWZVkCAAAAYMxlpgMAAAAALR2lHAAAADCMUg4AAAAYRikHAAAADKOUAwAAAIZRygEAAADDKOW4qMOHD+vGG2/UsGHDPP+GDh2qvLy8i26Xn5+vSZMmNVNK71RVVemBBx5QQUHBBcckJSU1eH9FRYWys7OVnp6uoUOHavjw4Vq1alXAspWVlWns2LEB219aWpp27NjRpH3k5eUpNTVVycnJmjFjhqqrqwOUDoA3mH/PaInzr+TdMUNocZoOAPu7/PLLtXbtWs/tsrIypaWlKT4+XnFxcQaTee/TTz/Vc889pwMHDmjMmDE+b//iiy+qdevWWrdunRwOh8rKyjRmzBjFxMRo4MCBTc4XFRWld999t8n7CZR9+/Zp/vz5Wr16tdq3b68pU6Zo6dKlevDBB01HA1oU5t+WN/9KTT9muDRRyuGzqKgoxcbGqqSkRHFxcVq0aJFWr14tp9Op2NhYzZ49u974zz77TDk5OaqqqpLL5dKAAQP0pz/9STU1NZo5c6Z27dql8PBwde3aVbNmzVJkZGSD97dp06befrOyslRaWlrvvq5du+rVV189L/Py5cv1xBNPaNGiRX49Z5fLpU6dOqm6uloRERGKiorS/Pnz1b59e0lnzvDMmzdPN910U73bHTp0UGZmpq6//np988036tu3r1q3bq3f//73kqStW7fqlVdeUW5urtLT0/XJJ58oKSlJr776quLj4yVJjz32mG699Vbde++9eu2117Rx40bV1dXpmmuu0YwZMxQVFaX9+/dr+vTpOn36tLp3765Tp041+Dy8PWabNm1SUlKSOnbsKEkaM2aM/vjHP1LKAcOYf0N//g3EMcOliVIOn3366ac6dOiQevfurU2bNik/P18rV65Uu3btNGvWLL311luKioryjF+2bJmysrJ022236ccff9Qdd9yh4uJiVVZWaufOndqwYYMcDodycnL01Vdfqa6ursH7+/btWy/Hyy+/7HXml156SZL8nuAeffRR/eY3v1G/fv10yy23qG/fvkpNTVW3bt0a3fbo0aN68cUX9Ytf/EL/+Mc/lJGRoaeffloRERFavXq1Ro8e7RkbFhamkSNHKj8/X/Hx8Tp58qS2bdummTNnas2aNdq3b59WrVolp9OpFStWKDs7W6+//rqmTJmizMxMZWRkqKioSJmZmQ1m8faYHTlyRF27dvXcjo6OVllZmVfbAgge5t/Qn3+lph8zXJoo5WhUZWWlhg0bJkmqra1Vhw4dlJOTo5iYGC1evFiDBw9Wu3btJEnTpk2TdGZN409mz56tDz/8UAsXLtSBAwfkdrt16tQpxcXFKSwsTBkZGRo4cKBSUlJ08803q7y8vMH7z+XLWYemiouLU0FBgfbu3au//e1v+vjjj7Vw4ULNmzfvgusgf+J0OtWnTx9JUrdu3XTDDTdo8+bN6t+/v7Zv367nn39e33//vWf8yJEjNWrUKE2dOlXr169XUlKSrrzySn3wwQf6/PPPNXLkSElSXV2dTp8+re+//15fffWVhg8fLklKSEjQz372swazeHvMLMs67/Zll/EWFKC5Mf+2vPkXLRelHI06d03j2cLCwuRwODy3y8vLVV5eXm/MfffdpxtuuEG333677r77bu3evVuWZalt27Zau3atdu3ape3bt+uxxx7Tr371K2VmZl7w/rP5ctahKWpqavTcc8/pt7/9reLj4xUfH68HHnhACxYs0IoVKzzfFM4uslVVVZ6PIyIi5HT+60tt9OjRWrNmjY4fP64777xTbdq0qfdN4ZprrlHPnj21ZcsW5efna/r06ZLOfBP49a9/rXvvvdfzGCdPnvRsd/bjn/14Z/P2mMXExOjYsWOe28eOHVN0dLRX2wIIHObfljf/ouXi1BeaZMCAAXr//fdVUVEhSZo/f76WLl3q+Xx5ebk+//xzTZkyRcnJyTp69KgOHTqkuro6ffDBB5owYYJuueUWTZ48WcOHD1dxcfEF7zfF6XTq4MGDWrBggecKJDU1Nfr666/Vs2dPSVLHjh09GXfs2CGXy3XB/d11113a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3qz+/ftr+/btev755/X99997xo8cOVKjRo3S1KlTtX79eiUlJenKK6/UBx98oM8//1wjR46UJNXV1en06dP6/vvv9dVXX2n48OGSpISEBP3sZz9rMIu3x8yyrPNuX3YZb0EBmhvzb8ubf9FyUcrRqHPXNJ4tLCxMDofDc7u8vFzl5eX1xtx333264YYbdPvtt+vuu+/W7t27ZVmW2rZtq7Vr12rXrl3avn27HnvsMf3qV79SZmbmBe8/m7dnHb788kvV1dWpZ8+ecjgcysjI0LJly7x+/jU1NXruuef029/+VvHx8YqPj9cDDzygBQsWaMWKFZ5vCmcX2aqqKs/HERERcjr/9aU2evRorVmzRsePH9edd96pNm3a1PumcM0116hnz57asmWL8vPzNX36dElnvgn8+te/1r333ut5jJMnT3q2O/vxz368s3l7zGJiYnTs2DHP7WPHjik6OtqrbQEEDvNvy5t/0XJx6gtNMmDAAL3//vuqqKiQJM2fP19Lly71fL68vFyff/65pkyZouTkZB09elSHDh1SXV2dPvjgA02YMEG33HKLJk+erOHDh6u4uPiC9/vryy+/1LRp0zzv+F+zZo369evn9fZOp1MHDx7UggULPFcgqamp0ddff62ePXtKkjp27OjJuGPHDrlcrgvu76677tLevXu1cuXKer86Pdvo0aP1+uuv6/Tp00pISJAkDRw4UHl5eZ5jPW/ePD311FPq0KGDevXq5bkawd69e7Vv3z6vn19DkpKStHnzZh0/flyWZWnFihW68847m7RPAIHF/Bua8y9aLs6Uo0kSExO1f/9+jRs3TpLUo0cPzZw5Uxs3bpQktW3bVhMnTtQ999yj1q1bKyoqSn379lVpaakyMjL04YcfKi0tTa1bt1a7du00c+ZMxcTENHi/v4YPH65Dhw5p5MiRCgsL089+9jM9//zzPu1j3rx5ysnJUUpKilq1aqW6ujrdddddeuSRRyRJU6ZM0bPPPqsVK1aoV69e6tWr1wX3FRERodTUVBUWFjb4a2HpTCn+wx/+UO+NlRkZGSorK9Po0aPlcDgUExPjeVPXSy+9pGnTpundd9/Vtddeq+7du/v0/M4VFxenRx55ROPHj1d1dbV69+7NmzwBm2H+Dc35Fy2Xwzp38SjQgv10hhgA0LyYf9HSsXwFAAAAMIwz5QAAAIBhnCkH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\n",
      "text/plain": [
       "<Figure size 746.08x691.2 with 6 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "### Graphe survie selon la classe et l'age\n",
    "grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=3.2, aspect=1.6)\n",
    "grid.map(plt.hist, 'Age', alpha=.5, bins=20, color=\"orange\")\n",
    "grid.add_legend();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Le graphique ci-dessus confirme notre hypothèse sur pclass 1, \n",
    "### mais nous pouvons également repérer une forte probabilité qu'une personne dans pclass 3 ne survive pas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Yes    537\n",
       "No     354\n",
       "Name: travelled_alone, dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Création des colonnes relative et travelled_alone :\n",
    "### On calcule le nombre de personnes qui accompagnent les passagers\n",
    "### Pour vérifier s'ils voyagent seuls ou accompagnés\n",
    "\n",
    "data = [train_data, test_data]\n",
    "for dataset in data:\n",
    "    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']\n",
    "    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'\n",
    "    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'\n",
    "    #dataset['travelled_alone'] = dataset['travelled_alone'].astype(int)\n",
    "train_data['travelled_alone'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Yes    253\n",
       "No     165\n",
       "Name: travelled_alone, dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher le nombre de passagers qui voyagent seuls ou accompagnés sur le test_data\n",
    "test_data['travelled_alone'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     537\n",
       "1     161\n",
       "2     102\n",
       "3      29\n",
       "5      22\n",
       "4      15\n",
       "6      12\n",
       "10      7\n",
       "7       6\n",
       "Name: relatives, dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher le nombre de passagers avec le nombre d'accompagnants\n",
    "train_data['relatives'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Troisième question qu'on s'est posé :\n",
    "### Les passagers ont-ils de meilleures chances de survie lorsqu'ils voyagent seuls?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\balde\\Anaconda3\\lib\\site-packages\\seaborn\\categorical.py:3666: UserWarning: The `factorplot` function has been renamed to `catplot`. The original name will be removed in a future release. Please update your code. Note that the default `kind` in `factorplot` (`'point'`) has changed `'strip'` in `catplot`.\n",
      "  warnings.warn(msg)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3gAAAFcCAYAAACazBxHAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nOzdd1xV9f8H8Ne9XPbeqAgiCMhSnLi3ljnKLJtmpb/KzPb8avX92jAbVmZDM7PSclTOcuLGAS4ERJYIyN7rcuf5/XHxKqWIyr3ncnk9Hw8feM+54+UA7otzzuctEQRBABEREREREbV5UrEDEBERERERUetgwSMiIiIiIjITLHhERERERERmggWPiIiIiIjITLDgERERERERmQmzKHiCIEChUIALghIRERERUXtmFgVPqVQiKSkJSqVS7ChERERERESiMYuCR0RERERERCx4REREREREZoMFj4iIiIiIyEyw4BEREREREZkJFjwiIiIiIiIzwYJHRERERERkJljwiIiIiIiIzAQLHhERERERkZlgwSMiIiIiIjITLHhERERERERmggWPiIiIiIjITLDgERGZifiUQrz59SHEpxSKHYWIiIhEIhM7ABERtY7VO1KRmVcFuUKNvmE+YschIiIiEfAIHhGRmZA3qJt8JCIiovaHBY+IiIiIiMhMsOARERERERGZCRY8IiIiIiIiM8GCR0REREREZCZY8IiIiIiIiMwECx4REREREZGZYMEjIiIiIiIyEyx4REREREREZoIFj4iIiIiIyEyw4BEREREREZkJFjwiIiIiIiIzwYJHRERERERkJljwiIiIiIiIzAQLHhERERERkZkwaMHbsmULxo8fj7Fjx2L16tX/2p+cnIx7770XkyZNwlNPPYXq6mpDxiEiIiIiIjJrBit4RUVFWLx4MdasWYONGzdi7dq1yMjIaHKf999/H3PnzsXmzZsREBCAFStWGCoOERERERGR2TNYwYuLi0NMTAxcXFxgZ2eHcePGYfv27U3uo9VqUVdXBwCQy+WwsbExVBwiIiIiIiKzJzPUExcXF8PT01N/28vLC4mJiU3u88Ybb+CJJ57ABx98AFtbW6xbt+62XjMpKem2Hk9E1JY1KBT6jydOnBA5DRERERlK7969r7vPYAVPq9VCIpHobwuC0OR2Q0MD/vOf/+DHH39EVFQUVq5ciddffx3Lli275deMiIiAtbX1beUmImqrbHbuBmrUsLG2bvYLPxEREZkvg52i6ePjg5KSEv3tkpISeHl56W+npaXB2toaUVFRAIBp06bh+PHjhopDRERERERk9gxW8AYOHIgjR46gvLwccrkcO3fuxNChQ/X7/f39UVhYiKysLADAnj17EBkZaag4REREREREZs9gp2h6e3vjxRdfxPTp06FSqTB16lRERUVh1qxZmDt3LiIjI/Hhhx/ihRdegCAIcHd3xwcffGCoOERERERERGZPIgiCIHaI26VQKJCUlMRr8IioXXvqw93IL61DRw97fPfmaLHjEBERkQgMOuiciIiIiIiIjIcFj4iIiIiIyEyw4BEREREREZkJFjwiIiIiIiIzwYJHRERERERkJljwiIiIiIiIzAQLHhERERERkZlgwSMiIiIiIjITLHhERERERERmggWPiIiIiIjITLDgERERERERmQkWPCIiIiIiIjPBgkdERERERGQmWPCIiMxAfkktauUqAIBSrYUgCCInIiIiIjGw4BERtWEqtRZf/HYKTy3cg+o6JQCgtFKOed/G6W8TERFR+8GCR0TUhv24NRm743P+tT0xoxQLV8XzSB4REVE7w4JHRNRG1dQrsf1I9nX3n80sRVpOhdHyEBERkfhY8IiI2qj0nEoo1dpm75OUWWakNERERGQKWPCIiNogjVZAUlbpDe9nYcEv80RERO2JTOwARETUcmqNFvtO5GH9njTkl9bd8P59unsZIRURERGZChY8IqI2QKXWYk98DjbEpqOovL5Fj+kf7gNfL0cDJyMiIiJTwoJHRGTClCoNdh27iA2x6SitatBvl0iAQVEdcd+objidVop1e9JQ1zgH7zIBXEGTiIiovWHBIyIyQQ0KNbYfzcYfezNQUaPQb5dKgKG9fHH/qGB09tYdnevayQXjB3XB0wv3oKyqARZSCTRaAceTi3AmvQQ9unmK9ccgIiIiI2PBIyIyIfUNKmw7fAGbDmSiqvbKoHILqQQj+3TG1FHd0NHD4V+Ps7GSwdrSAgDg7GCN8mrd0b7vNyXh85eGw0IqMc4fgIiIiETFgkdEZAJq5SpsOZiFzQcyUXvVqZYyCynG9PPDvSO7wdvNrkXPZWNlgV6hXjiZWozsgmrsPn4R42K6GCg5ERERmRIWPCIiEVXX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\n",
      "text/plain": [
       "<Figure size 900x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "### Graphe survie selon le nombre de personnes d'accompagnants\n",
    "axes = sns.factorplot('relatives','Survived', \n",
    "                      data=train_data, aspect = 2.5, );"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Traitement des données"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Supprimer la colonne passengerId\n",
    "train_data = train_data.drop(['PassengerId'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count             204\n",
       "unique            147\n",
       "top       C23 C25 C27\n",
       "freq                4\n",
       "Name: Cabin, dtype: object"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher les détails statistique de la colonne cabine\n",
    "train_data['Cabin'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Traitement de la colonne cabin\n",
    "### Regrouper les valeurs par caractère \n",
    "### remplacer les valeurs manquantes par U\n",
    "### Créer une novelle colonnes Deck à partir de la colonne cabin\n",
    "import re\n",
    "deck = {\"A\": \"A\", \"B\": \"B\", \"C\": \"C\", \"D\": \"D\", \"E\": \"E\", \"F\": \"F\", \"G\": \"G\", \"U\": \"U\"}\n",
    "data = [train_data, test_data]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Cabin'] = dataset['Cabin'].fillna(\"U0\")\n",
    "    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile(\"([a-zA-Z]+)\").search(x).group())\n",
    "    dataset['Deck'] = dataset['Deck'].map(deck)\n",
    "    dataset['Deck'] = dataset['Deck'].fillna(\"U\")\n",
    "    \n",
    "### Supprimer la colonne Cabin\n",
    "train_data = train_data.drop(['Cabin'], axis=1)\n",
    "test_data = test_data.drop(['Cabin'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "U    688\n",
       "C     59\n",
       "B     47\n",
       "D     33\n",
       "E     32\n",
       "A     15\n",
       "F     13\n",
       "G      4\n",
       "Name: Deck, dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher le nombre de passagers par Deck\n",
    "train_data['Deck'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>relatives</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Deck</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>A</th>\n",
       "      <td>0.466667</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>44.833333</td>\n",
       "      <td>0.133333</td>\n",
       "      <td>0.133333</td>\n",
       "      <td>39.623887</td>\n",
       "      <td>0.266667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>B</th>\n",
       "      <td>0.744681</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>34.955556</td>\n",
       "      <td>0.361702</td>\n",
       "      <td>0.574468</td>\n",
       "      <td>113.505764</td>\n",
       "      <td>0.936170</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>C</th>\n",
       "      <td>0.593220</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>36.086667</td>\n",
       "      <td>0.644068</td>\n",
       "      <td>0.474576</td>\n",
       "      <td>100.151341</td>\n",
       "      <td>1.118644</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>D</th>\n",
       "      <td>0.757576</td>\n",
       "      <td>1.121212</td>\n",
       "      <td>39.032258</td>\n",
       "      <td>0.424242</td>\n",
       "      <td>0.303030</td>\n",
       "      <td>57.244576</td>\n",
       "      <td>0.727273</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>E</th>\n",
       "      <td>0.750000</td>\n",
       "      <td>1.312500</td>\n",
       "      <td>38.116667</td>\n",
       "      <td>0.312500</td>\n",
       "      <td>0.312500</td>\n",
       "      <td>46.026694</td>\n",
       "      <td>0.625000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>F</th>\n",
       "      <td>0.615385</td>\n",
       "      <td>2.384615</td>\n",
       "      <td>19.954545</td>\n",
       "      <td>0.538462</td>\n",
       "      <td>0.538462</td>\n",
       "      <td>18.696792</td>\n",
       "      <td>1.076923</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>G</th>\n",
       "      <td>0.500000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>14.750000</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>1.250000</td>\n",
       "      <td>13.581250</td>\n",
       "      <td>1.750000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>U</th>\n",
       "      <td>0.299419</td>\n",
       "      <td>2.636628</td>\n",
       "      <td>27.588208</td>\n",
       "      <td>0.546512</td>\n",
       "      <td>0.364826</td>\n",
       "      <td>19.181079</td>\n",
       "      <td>0.911337</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Survived    Pclass        Age     SibSp     Parch        Fare  relatives\n",
       "Deck                                                                          \n",
       "A     0.466667  1.000000  44.833333  0.133333  0.133333   39.623887   0.266667\n",
       "B     0.744681  1.000000  34.955556  0.361702  0.574468  113.505764   0.936170\n",
       "C     0.593220  1.000000  36.086667  0.644068  0.474576  100.151341   1.118644\n",
       "D     0.757576  1.121212  39.032258  0.424242  0.303030   57.244576   0.727273\n",
       "E     0.750000  1.312500  38.116667  0.312500  0.312500   46.026694   0.625000\n",
       "F     0.615385  2.384615  19.954545  0.538462  0.538462   18.696792   1.076923\n",
       "G     0.500000  3.000000  14.750000  0.500000  1.250000   13.581250   1.750000\n",
       "U     0.299419  2.636628  27.588208  0.546512  0.364826   19.181079   0.911337"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Regroupement par moyenne et par Deck\n",
    "train_data.groupby('Deck').mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "U    327\n",
       "C     35\n",
       "B     18\n",
       "D     13\n",
       "E      9\n",
       "F      8\n",
       "A      7\n",
       "G      1\n",
       "Name: Deck, dtype: int64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher le nombre de passagers par Deck dans le fichier test_data\n",
    "test_data['Deck'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Traitement de la colonne age\n",
    "### Calculer la moyenne et la variance de la colonne age\n",
    "### Remplacer les valeurs vides par des nombres aléatoires calculés à partir de la moyenne et la variance\n",
    "\n",
    "data = [train_data, test_data]\n",
    "\n",
    "for dataset in data:\n",
    "    mean = train_data[\"Age\"].mean()\n",
    "    std = test_data[\"Age\"].std()\n",
    "    is_null = dataset[\"Age\"].isnull().sum()\n",
    "    rand_age = np.random.randint(mean - std, mean + std, size = is_null)\n",
    "    age_slice = dataset[\"Age\"].copy()\n",
    "    age_slice[np.isnan(age_slice)] = rand_age\n",
    "    dataset[\"Age\"] = age_slice\n",
    "    dataset[\"Age\"] = train_data[\"Age\"].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Vérification\n",
    "train_data[\"Age\"].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    891.000000\n",
       "mean      29.283951\n",
       "std       13.549044\n",
       "min        0.000000\n",
       "25%       20.000000\n",
       "50%       28.000000\n",
       "75%       37.000000\n",
       "max       80.000000\n",
       "Name: Age, dtype: float64"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher les détails statistique de la colonne Age\n",
    "\n",
    "train_data[\"Age\"].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     889\n",
       "unique      3\n",
       "top         S\n",
       "freq      644\n",
       "Name: Embarked, dtype: object"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher les détails statistique de la colonne Embarked\n",
    "\n",
    "train_data['Embarked'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    S\n",
       "dtype: object"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher le port d'embarquement avec la majorité de données\n",
    "\n",
    "train_data['Embarked'].mode()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Remplacer les valeurs vides par le port S\n",
    "### Nous supposons que les passagers qui n'ont pas de port mentionné se sont embarqués au port S\n",
    "common_value = 'S'\n",
    "data = [train_data, test_data]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     418\n",
       "unique      3\n",
       "top         S\n",
       "freq      270\n",
       "Name: Embarked, dtype: object"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher les détails statistique de la colonne Embarked\n",
    "test_data['Embarked'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Convertir la colonnes Fare en int\n",
    "data = [train_data, test_data]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Fare'] = dataset['Fare'].fillna(0)\n",
    "    dataset['Fare'] = dataset['Fare'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    891.000000\n",
       "mean      31.785634\n",
       "std       49.703730\n",
       "min        0.000000\n",
       "25%        7.000000\n",
       "50%       14.000000\n",
       "75%       31.000000\n",
       "max      512.000000\n",
       "Name: Fare, dtype: float64"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher les détails statistique de la colonne Fare\n",
    "train_data['Fare'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Vérifier si on a des valeurs vides dans la colonne Fare\n",
    "train_data['Fare'].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    418.000000\n",
       "mean      35.100478\n",
       "std       55.872752\n",
       "min        0.000000\n",
       "25%        7.000000\n",
       "50%       14.000000\n",
       "75%       31.000000\n",
       "max      512.000000\n",
       "Name: Fare, dtype: float64"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher les détails statistique de la colonne Fare\n",
    "test_data['Fare'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Vérifier si on a des valeurs vides dans la colonne Fare fichier test\n",
    "test_data['Fare'].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Convertir en int fichier train\n",
    "train_data['Fare'] = train_data['Fare'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Convertir en int sur le fichier test\n",
    "test_data['Fare'] = test_data['Fare'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Traitement de la colonne Name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pandas.core.series.Series"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Extraire les titres des passagers\n",
    "train_titles = train_data.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)\n",
    "type(train_titles)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Mr          517\n",
       "Miss        182\n",
       "Mrs         125\n",
       "Master       40\n",
       "Dr            7\n",
       "Rev           6\n",
       "Col           2\n",
       "Major         2\n",
       "Mlle          2\n",
       "Ms            1\n",
       "Countess      1\n",
       "Lady          1\n",
       "Don           1\n",
       "Jonkheer      1\n",
       "Capt          1\n",
       "Sir           1\n",
       "Mme           1\n",
       "Name: Name, dtype: int64"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Afficher le nombre de passagers par titre\n",
    "train_titles.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Remplacer les titres par des entiers pour faciliter le traitement\n",
    "### Remplacer les titres non fréquent par rare\n",
    "### Remplacer: Mlle par Miss, Ms par Miss, Mme par Mrs\n",
    "### Ceci pour une meilleure catégorisation des données\n",
    "\n",
    "data = [train_data, test_data]\n",
    "titles = {\"Mr\": 1, \"Miss\": 2, \"Mrs\": 3, \"Master\": 4, \"Rare\": 5}\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)\n",
    "    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\\\n",
    "                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')\n",
    "    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')\n",
    "    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')\n",
    "    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')\n",
    "train_data = train_data.drop(['Name'], axis=1)\n",
    "test_data = test_data.drop(['Name'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>relatives</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Master</th>\n",
       "      <td>0.575000</td>\n",
       "      <td>2.625000</td>\n",
       "      <td>6.950000</td>\n",
       "      <td>2.300000</td>\n",
       "      <td>1.375000</td>\n",
       "      <td>34.250000</td>\n",
       "      <td>3.675000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Miss</th>\n",
       "      <td>0.702703</td>\n",
       "      <td>2.291892</td>\n",
       "      <td>23.205405</td>\n",
       "      <td>0.702703</td>\n",
       "      <td>0.540541</td>\n",
       "      <td>43.340541</td>\n",
       "      <td>1.243243</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mr</th>\n",
       "      <td>0.156673</td>\n",
       "      <td>2.410058</td>\n",
       "      <td>31.297872</td>\n",
       "      <td>0.288201</td>\n",
       "      <td>0.152805</td>\n",
       "      <td>24.021277</td>\n",
       "      <td>0.441006</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mrs</th>\n",
       "      <td>0.793651</td>\n",
       "      <td>1.992063</td>\n",
       "      <td>34.182540</td>\n",
       "      <td>0.690476</td>\n",
       "      <td>0.825397</td>\n",
       "      <td>44.984127</td>\n",
       "      <td>1.515873</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Rare</th>\n",
       "      <td>0.347826</td>\n",
       "      <td>1.347826</td>\n",
       "      <td>44.913043</td>\n",
       "      <td>0.347826</td>\n",
       "      <td>0.086957</td>\n",
       "      <td>36.782609</td>\n",
       "      <td>0.434783</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Survived    Pclass        Age     SibSp     Parch       Fare  \\\n",
       "Title                                                                  \n",
       "Master  0.575000  2.625000   6.950000  2.300000  1.375000  34.250000   \n",
       "Miss    0.702703  2.291892  23.205405  0.702703  0.540541  43.340541   \n",
       "Mr      0.156673  2.410058  31.297872  0.288201  0.152805  24.021277   \n",
       "Mrs     0.793651  1.992063  34.182540  0.690476  0.825397  44.984127   \n",
       "Rare    0.347826  1.347826  44.913043  0.347826  0.086957  36.782609   \n",
       "\n",
       "        relatives  \n",
       "Title              \n",
       "Master   3.675000  \n",
       "Miss     1.243243  \n",
       "Mr       0.441006  \n",
       "Mrs      1.515873  \n",
       "Rare     0.434783  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Repartition de la moyenne par titre\n",
    "train_data.groupby(['Title']).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "#### Traitement de la colonne sexe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "male      577\n",
       "female    314\n",
       "Name: Sex, dtype: int64"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Repartition des données par sexe\n",
    "train_data['Sex'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\n### Remplacer les caractère par des entiers\\ngenre = {\"male\": 0, \"female\": 1}\\ndata = [train_data, test_data]\\n\\nfor dataset in data:\\n    dataset[\\'Sex\\'] = dataset[\\'Sex\\'].map(genre)\\n'"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "### Remplacer les caractère par des entiers\n",
    "genre = {\"male\": 0, \"female\": 1}\n",
    "data = [train_data, test_data]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Sex'] = dataset['Sex'].map(genre)\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "male      577\n",
       "female    314\n",
       "Name: Sex, dtype: int64"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Vérification\n",
    "train_data['Sex'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count        891\n",
       "unique       681\n",
       "top       347082\n",
       "freq           7\n",
       "Name: Ticket, dtype: object"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "###Traitement de la colonne Ticket\n",
    "train_data['Ticket'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Cette colonne contient 681 valeurs uniques, il n'est donc pas pertinent de travailler dessus.Nous allons donc supprimer cette colonne.\n",
    "train_data = train_data.drop(['Ticket'], axis=1)\n",
    "test_data = test_data.drop(['Ticket'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'\\nports = {\"S\": 0, \"C\": 1, \"Q\": 2}\\ndata = [train_df, test_df]\\n\\nfor dataset in data:\\n    dataset[\\'Embarked\\'] = dataset[\\'Embarked\\'].map(ports)\\n'"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "'''\n",
    "ports = {\"S\": 0, \"C\": 1, \"Q\": 2}\n",
    "data = [train_df, test_df]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Embarked'] = dataset['Embarked'].map(ports)\n",
    "'''    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "### Traitement sur la colonne Age\n",
    "### Dans une nouvelle colonne, nous allons calculer un indicateur en multipliant la classe et l'age du passager\n",
    "data = [train_data, test_data] \n",
    "for dataset in data:\n",
    "    dataset['Age_Class']= dataset['Age']* dataset['Pclass']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Après analyse sur l'age, nous decidons de classer les passagers en 8 catégories d'age\n",
    "data = [train_data, test_data]\n",
    "for dataset in data:\n",
    "    dataset['Age'] = dataset['Age'].astype(int)\n",
    "    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0\n",
    "    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1\n",
    "    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2\n",
    "    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3\n",
    "    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4\n",
    "    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5\n",
    "    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6\n",
    "    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7\n",
    "    \n",
    "    dataset['Age'] = dataset['Age'].astype(str)\n",
    "    dataset.loc[ dataset['Age'] == '0', 'Age'] = \"Children\"\n",
    "    dataset.loc[ dataset['Age'] == '1', 'Age'] = \"Teens\"\n",
    "    dataset.loc[ dataset['Age'] == '2', 'Age'] = \"Youngsters\"\n",
    "    dataset.loc[ dataset['Age'] == '3', 'Age'] = \"Young Adults\"\n",
    "    dataset.loc[ dataset['Age'] == '4', 'Age'] = \"Adults\"\n",
    "    dataset.loc[ dataset['Age'] == '5', 'Age'] = \"Middle Age\"\n",
    "    dataset.loc[ dataset['Age'] == '6', 'Age'] = \"Senior\"\n",
    "    dataset.loc[ dataset['Age'] == '7', 'Age'] = \"Retired\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Après analyse des prix, nous supposons que le prix affiché correspond au montant total payé par le voyageur, qu'il soit acccompagné ou seul\n",
    "### Il est donc necessaire de calculer le prix par passager en divisant le prix par le nombre de personne \n",
    "for dataset in data:\n",
    "    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)\n",
    "    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "##test_data['Age'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 14 columns):\n",
      "Survived           891 non-null int64\n",
      "Pclass             891 non-null int64\n",
      "Sex                891 non-null object\n",
      "Age                891 non-null object\n",
      "SibSp              891 non-null int64\n",
      "Parch              891 non-null int64\n",
      "Fare               891 non-null int32\n",
      "Embarked           891 non-null object\n",
      "relatives          891 non-null int64\n",
      "travelled_alone    891 non-null object\n",
      "Deck               891 non-null object\n",
      "Title              891 non-null object\n",
      "Age_Class          891 non-null int64\n",
      "Fare_Per_Person    891 non-null int32\n",
      "dtypes: int32(2), int64(6), object(6)\n",
      "memory usage: 90.6+ KB\n"
     ]
    }
   ],
   "source": [
    "### Afficher les informations sur notre fichier train\n",
    "train_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Traitement de la colonne Fare (Prix)\n",
    "### Après une analyse approfondie(Dispersion et repartition) nous decidons de repartir les prix en 6 catégories \n",
    "data = [train_data, test_data]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0\n",
    "    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1\n",
    "    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2\n",
    "    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3\n",
    "    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4\n",
    "    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5\n",
    "    dataset['Fare'] = dataset['Fare'].astype(int)\n",
    "    \n",
    "    dataset['Fare'] = dataset['Fare'].astype(str)\n",
    "    dataset.loc[ dataset['Fare'] == '0', 'Fare'] = \"Extremely Low\"\n",
    "    dataset.loc[ dataset['Fare'] == '1', 'Fare'] = \"Very Low\"\n",
    "    dataset.loc[ dataset['Fare'] == '2', 'Fare'] = \"Low\"\n",
    "    dataset.loc[ dataset['Fare'] == '3', 'Fare'] = \"High\"\n",
    "    dataset.loc[ dataset['Fare'] == '4', 'Fare'] = \"Very High\"\n",
    "    dataset.loc[ dataset['Fare'] == '5', 'Fare'] = \"Extremely High\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Extremely Low     241\n",
       "Low               223\n",
       "Very Low          216\n",
       "High              158\n",
       "Very High          44\n",
       "Extremely High      9\n",
       "Name: Fare, dtype: int64"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Vérifications\n",
    "train_data['Fare'].value_counts()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Extremely Low     120\n",
       "Low               102\n",
       "Very Low           96\n",
       "High               69\n",
       "Very High          23\n",
       "Extremely High      8\n",
       "Name: Fare, dtype: int64"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data['Fare'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 14 columns):\n",
      "Survived           891 non-null int64\n",
      "Pclass             891 non-null int64\n",
      "Sex                891 non-null object\n",
      "Age                891 non-null object\n",
      "SibSp              891 non-null int64\n",
      "Parch              891 non-null int64\n",
      "Fare               891 non-null object\n",
      "Embarked           891 non-null object\n",
      "relatives          891 non-null int64\n",
      "travelled_alone    891 non-null object\n",
      "Deck               891 non-null object\n",
      "Title              891 non-null object\n",
      "Age_Class          891 non-null int64\n",
      "Fare_Per_Person    891 non-null int32\n",
      "dtypes: int32(1), int64(6), object(7)\n",
      "memory usage: 94.1+ KB\n"
     ]
    }
   ],
   "source": [
    "train_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 418 entries, 0 to 417\n",
      "Data columns (total 14 columns):\n",
      "PassengerId        418 non-null int64\n",
      "Pclass             418 non-null int64\n",
      "Sex                418 non-null object\n",
      "Age                418 non-null object\n",
      "SibSp              418 non-null int64\n",
      "Parch              418 non-null int64\n",
      "Fare               418 non-null object\n",
      "Embarked           418 non-null object\n",
      "relatives          418 non-null int64\n",
      "travelled_alone    418 non-null object\n",
      "Deck               418 non-null object\n",
      "Title              418 non-null object\n",
      "Age_Class          418 non-null int64\n",
      "Fare_Per_Person    418 non-null int32\n",
      "dtypes: int32(1), int64(6), object(7)\n",
      "memory usage: 44.2+ KB\n"
     ]
    }
   ],
   "source": [
    "test_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>relatives</th>\n",
       "      <th>travelled_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "      <th>Age_Class</th>\n",
       "      <th>Fare_Per_Person</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>Youngsters</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Extremely Low</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>66</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>High</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>38</td>\n",
       "      <td>35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>Young Adults</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Extremely Low</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Miss</td>\n",
       "      <td>78</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>High</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>35</td>\n",
       "      <td>26</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>105</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>Youngsters</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>Q</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>63</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>Senior</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>High</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>E</td>\n",
       "      <td>Mr</td>\n",
       "      <td>54</td>\n",
       "      <td>51</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>Children</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>Low</td>\n",
       "      <td>S</td>\n",
       "      <td>4</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Master</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>Young Adults</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>S</td>\n",
       "      <td>2</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>81</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>female</td>\n",
       "      <td>Teens</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Low</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>28</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex           Age  SibSp  Parch           Fare  \\\n",
       "0         0       3    male    Youngsters      1      0  Extremely Low   \n",
       "1         1       1  female    Middle Age      1      0           High   \n",
       "2         1       3  female  Young Adults      0      0  Extremely Low   \n",
       "3         1       1  female    Middle Age      1      0           High   \n",
       "4         0       3    male    Middle Age      0      0       Very Low   \n",
       "5         0       3    male    Youngsters      0      0       Very Low   \n",
       "6         0       1    male        Senior      0      0           High   \n",
       "7         0       3    male      Children      3      1            Low   \n",
       "8         1       3  female  Young Adults      0      2       Very Low   \n",
       "9         1       2  female         Teens      1      0            Low   \n",
       "\n",
       "  Embarked  relatives travelled_alone Deck   Title  Age_Class  Fare_Per_Person  \n",
       "0        S          1              No    U      Mr         66                3  \n",
       "1        C          1              No    C     Mrs         38               35  \n",
       "2        S          0             Yes    U    Miss         78                7  \n",
       "3        S          1              No    C     Mrs         35               26  \n",
       "4        S          0             Yes    U      Mr        105                8  \n",
       "5        Q          0             Yes    U      Mr         63                8  \n",
       "6        S          0             Yes    E      Mr         54               51  \n",
       "7        S          4              No    U  Master          6                4  \n",
       "8        S          2              No    U     Mrs         81                3  \n",
       "9        C          1              No    U     Mrs         28               15  "
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3    491\n",
       "1    216\n",
       "2    184\n",
       "Name: Pclass, dtype: int64"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Taitement de la colonne Pclass\n",
    "### Affichage de la repartition des données par Pclass\n",
    "train_data['Pclass'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Pour la suite nous allons modifier la colonne Pclass, en remplaçant les entiers par une chaine de caractère\n",
    "data = [train_data, test_data]\n",
    "\n",
    "for dataset in data:\n",
    "    dataset['Pclass'] = dataset['Pclass'].astype(str)\n",
    "    dataset.loc[ dataset['Pclass'] == '1', 'Pclass'] = \"Class1\"\n",
    "    dataset.loc[ dataset['Pclass'] == '2', 'Pclass'] = \"Class2\"\n",
    "    dataset.loc[ dataset['Pclass'] == '3', 'Pclass'] = \"Class3\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 14 columns):\n",
      "Survived           891 non-null int64\n",
      "Pclass             891 non-null object\n",
      "Sex                891 non-null object\n",
      "Age                891 non-null object\n",
      "SibSp              891 non-null int64\n",
      "Parch              891 non-null int64\n",
      "Fare               891 non-null object\n",
      "Embarked           891 non-null object\n",
      "relatives          891 non-null int64\n",
      "travelled_alone    891 non-null object\n",
      "Deck               891 non-null object\n",
      "Title              891 non-null object\n",
      "Age_Class          891 non-null int64\n",
      "Fare_Per_Person    891 non-null int32\n",
      "dtypes: int32(1), int64(5), object(8)\n",
      "memory usage: 94.1+ KB\n"
     ]
    }
   ],
   "source": [
    "### vérification\n",
    "train_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>relatives</th>\n",
       "      <th>travelled_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "      <th>Age_Class</th>\n",
       "      <th>Fare_Per_Person</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>Class3</td>\n",
       "      <td>male</td>\n",
       "      <td>Youngsters</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Extremely Low</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>66</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Class1</td>\n",
       "      <td>female</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>High</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>38</td>\n",
       "      <td>35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>Class3</td>\n",
       "      <td>female</td>\n",
       "      <td>Young Adults</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Extremely Low</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Miss</td>\n",
       "      <td>78</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>Class1</td>\n",
       "      <td>female</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>High</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>35</td>\n",
       "      <td>26</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>Class3</td>\n",
       "      <td>male</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>105</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>Class3</td>\n",
       "      <td>male</td>\n",
       "      <td>Youngsters</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>Q</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>63</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0</td>\n",
       "      <td>Class1</td>\n",
       "      <td>male</td>\n",
       "      <td>Senior</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>High</td>\n",
       "      <td>S</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>E</td>\n",
       "      <td>Mr</td>\n",
       "      <td>54</td>\n",
       "      <td>51</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0</td>\n",
       "      <td>Class3</td>\n",
       "      <td>male</td>\n",
       "      <td>Children</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>Low</td>\n",
       "      <td>S</td>\n",
       "      <td>4</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Master</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1</td>\n",
       "      <td>Class3</td>\n",
       "      <td>female</td>\n",
       "      <td>Young Adults</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>S</td>\n",
       "      <td>2</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>81</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1</td>\n",
       "      <td>Class2</td>\n",
       "      <td>female</td>\n",
       "      <td>Teens</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Low</td>\n",
       "      <td>C</td>\n",
       "      <td>1</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>28</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex           Age  SibSp  Parch           Fare  \\\n",
       "0         0  Class3    male    Youngsters      1      0  Extremely Low   \n",
       "1         1  Class1  female    Middle Age      1      0           High   \n",
       "2         1  Class3  female  Young Adults      0      0  Extremely Low   \n",
       "3         1  Class1  female    Middle Age      1      0           High   \n",
       "4         0  Class3    male    Middle Age      0      0       Very Low   \n",
       "5         0  Class3    male    Youngsters      0      0       Very Low   \n",
       "6         0  Class1    male        Senior      0      0           High   \n",
       "7         0  Class3    male      Children      3      1            Low   \n",
       "8         1  Class3  female  Young Adults      0      2       Very Low   \n",
       "9         1  Class2  female         Teens      1      0            Low   \n",
       "\n",
       "  Embarked  relatives travelled_alone Deck   Title  Age_Class  Fare_Per_Person  \n",
       "0        S          1              No    U      Mr         66                3  \n",
       "1        C          1              No    C     Mrs         38               35  \n",
       "2        S          0             Yes    U    Miss         78                7  \n",
       "3        S          1              No    C     Mrs         35               26  \n",
       "4        S          0             Yes    U      Mr        105                8  \n",
       "5        Q          0             Yes    U      Mr         63                8  \n",
       "6        S          0             Yes    E      Mr         54               51  \n",
       "7        S          4              No    U  Master          6                4  \n",
       "8        S          2              No    U     Mrs         81                3  \n",
       "9        C          1              No    U     Mrs         28               15  "
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Class3    491\n",
       "Class1    216\n",
       "Class2    184\n",
       "Name: Pclass, dtype: int64"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data['Pclass'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "### La mise à l’echelle\n",
    "### Sélection des données numériques, puis mise à l'échelle par la technique StandardScaler() (rapport moyenne et ecart type)\n",
    "### Ce traitement sera effectué sur les deux fichiers (train et test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Survived', 'SibSp', 'Parch', 'relatives', 'Age_Class', 'Fare_Per_Person']"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Train\n",
    "train_numerical_features = list(train_data.select_dtypes(include=['int64', 'float64', 'int32']).columns)\n",
    "train_numerical_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "list"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(train_numerical_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['SibSp', 'Parch', 'relatives', 'Age_Class', 'Fare_Per_Person']"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "del train_numerical_features[0]\n",
    "train_numerical_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Pour une mise à l'échelle moins risquée nous avons choisi la technique StandardScaler()\n",
    "ss_scaler = StandardScaler()\n",
    "train_data_ss = pd.DataFrame(data = train_data)\n",
    "train_data_ss[train_numerical_features] = ss_scaler.fit_transform(train_data_ss[train_numerical_features])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 14)"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Vérifications\n",
    "train_data_ss.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>relatives</th>\n",
       "      <th>travelled_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "      <th>Age_Class</th>\n",
       "      <th>Fare_Per_Person</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>Class3</td>\n",
       "      <td>male</td>\n",
       "      <td>Youngsters</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>Extremely Low</td>\n",
       "      <td>S</td>\n",
       "      <td>0.059160</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>0.059858</td>\n",
       "      <td>-0.459218</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>Class1</td>\n",
       "      <td>female</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>High</td>\n",
       "      <td>C</td>\n",
       "      <td>0.059160</td>\n",
       "      <td>No</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>-0.762913</td>\n",
       "      <td>0.434090</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>Class3</td>\n",
       "      <td>female</td>\n",
       "      <td>Young Adults</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>Extremely Low</td>\n",
       "      <td>S</td>\n",
       "      <td>-0.560975</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Miss</td>\n",
       "      <td>0.412473</td>\n",
       "      <td>-0.347554</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>Class1</td>\n",
       "      <td>female</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>High</td>\n",
       "      <td>S</td>\n",
       "      <td>0.059160</td>\n",
       "      <td>No</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>-0.851067</td>\n",
       "      <td>0.182847</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>Class3</td>\n",
       "      <td>male</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>S</td>\n",
       "      <td>-0.560975</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>1.205859</td>\n",
       "      <td>-0.319638</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex           Age     SibSp     Parch           Fare  \\\n",
       "0         0  Class3    male    Youngsters  0.432793 -0.473674  Extremely Low   \n",
       "1         1  Class1  female    Middle Age  0.432793 -0.473674           High   \n",
       "2         1  Class3  female  Young Adults -0.474545 -0.473674  Extremely Low   \n",
       "3         1  Class1  female    Middle Age  0.432793 -0.473674           High   \n",
       "4         0  Class3    male    Middle Age -0.474545 -0.473674       Very Low   \n",
       "\n",
       "  Embarked  relatives travelled_alone Deck Title  Age_Class  Fare_Per_Person  \n",
       "0        S   0.059160              No    U    Mr   0.059858        -0.459218  \n",
       "1        C   0.059160              No    C   Mrs  -0.762913         0.434090  \n",
       "2        S  -0.560975             Yes    U  Miss   0.412473        -0.347554  \n",
       "3        S   0.059160              No    C   Mrs  -0.851067         0.182847  \n",
       "4        S  -0.560975             Yes    U    Mr   1.205859        -0.319638  "
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data_ss.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['PassengerId', 'SibSp', 'Parch', 'relatives', 'Age_Class', 'Fare_Per_Person']"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Test\n",
    "test_numerical_features = list(test_data.select_dtypes(include=['int64', 'float64', 'int32']).columns)\n",
    "test_numerical_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['SibSp', 'Parch', 'relatives', 'Age_Class', 'Fare_Per_Person']"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "del test_numerical_features[0]\n",
    "test_numerical_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_ss_scaler = StandardScaler()\n",
    "test_data_ss = pd.DataFrame(data = test_data)\n",
    "test_data_ss[test_numerical_features] = test_ss_scaler.fit_transform(test_data_ss[test_numerical_features])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(418, 14)"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### Vérifications\n",
    "test_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>relatives</th>\n",
       "      <th>travelled_alone</th>\n",
       "      <th>Deck</th>\n",
       "      <th>Title</th>\n",
       "      <th>Age_Class</th>\n",
       "      <th>Fare_Per_Person</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>Class3</td>\n",
       "      <td>male</td>\n",
       "      <td>Youngsters</td>\n",
       "      <td>-0.499470</td>\n",
       "      <td>-0.400248</td>\n",
       "      <td>Extremely Low</td>\n",
       "      <td>Q</td>\n",
       "      <td>-0.553443</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>0.027878</td>\n",
       "      <td>-0.401204</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>Class3</td>\n",
       "      <td>female</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>0.616992</td>\n",
       "      <td>-0.400248</td>\n",
       "      <td>Extremely Low</td>\n",
       "      <td>S</td>\n",
       "      <td>0.105643</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>1.228201</td>\n",
       "      <td>-0.513662</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>Class2</td>\n",
       "      <td>male</td>\n",
       "      <td>Young Adults</td>\n",
       "      <td>-0.499470</td>\n",
       "      <td>-0.400248</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>Q</td>\n",
       "      <td>-0.553443</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>-0.322216</td>\n",
       "      <td>-0.344975</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>Class3</td>\n",
       "      <td>male</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>-0.499470</td>\n",
       "      <td>-0.400248</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>S</td>\n",
       "      <td>-0.553443</td>\n",
       "      <td>Yes</td>\n",
       "      <td>U</td>\n",
       "      <td>Mr</td>\n",
       "      <td>1.003140</td>\n",
       "      <td>-0.373089</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>Class3</td>\n",
       "      <td>female</td>\n",
       "      <td>Middle Age</td>\n",
       "      <td>0.616992</td>\n",
       "      <td>0.619896</td>\n",
       "      <td>Very Low</td>\n",
       "      <td>S</td>\n",
       "      <td>0.764728</td>\n",
       "      <td>No</td>\n",
       "      <td>U</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>1.003140</td>\n",
       "      <td>-0.485547</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass     Sex           Age     SibSp     Parch  \\\n",
       "0          892  Class3    male    Youngsters -0.499470 -0.400248   \n",
       "1          893  Class3  female    Middle Age  0.616992 -0.400248   \n",
       "2          894  Class2    male  Young Adults -0.499470 -0.400248   \n",
       "3          895  Class3    male    Middle Age -0.499470 -0.400248   \n",
       "4          896  Class3  female    Middle Age  0.616992  0.619896   \n",
       "\n",
       "            Fare Embarked  relatives travelled_alone Deck Title  Age_Class  \\\n",
       "0  Extremely Low        Q  -0.553443             Yes    U    Mr   0.027878   \n",
       "1  Extremely Low        S   0.105643              No    U   Mrs   1.228201   \n",
       "2       Very Low        Q  -0.553443             Yes    U    Mr  -0.322216   \n",
       "3       Very Low        S  -0.553443             Yes    U    Mr   1.003140   \n",
       "4       Very Low        S   0.764728              No    U   Mrs   1.003140   \n",
       "\n",
       "   Fare_Per_Person  \n",
       "0        -0.401204  \n",
       "1        -0.513662  \n",
       "2        -0.344975  \n",
       "3        -0.373089  \n",
       "4        -0.485547  "
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Encodage Binaire (création des variables catégorielles)\n",
    "### Pour ce faire nous allons utiliser la fonction dummies()\n",
    "### Train\n",
    "encode_col_list = list(train_data.select_dtypes(include=['object']).columns)\n",
    "for i in encode_col_list:\n",
    "    train_data_ss = pd.concat([train_data_ss,pd.get_dummies(train_data_ss[i], prefix=i)],axis=1)\n",
    "    train_data_ss.drop(i, axis = 1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 43)"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data_ss.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>relatives</th>\n",
       "      <th>Age_Class</th>\n",
       "      <th>Fare_Per_Person</th>\n",
       "      <th>Pclass_Class1</th>\n",
       "      <th>Pclass_Class2</th>\n",
       "      <th>Pclass_Class3</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>...</th>\n",
       "      <th>Deck_D</th>\n",
       "      <th>Deck_E</th>\n",
       "      <th>Deck_F</th>\n",
       "      <th>Deck_G</th>\n",
       "      <th>Deck_U</th>\n",
       "      <th>Title_Master</th>\n",
       "      <th>Title_Miss</th>\n",
       "      <th>Title_Mr</th>\n",
       "      <th>Title_Mrs</th>\n",
       "      <th>Title_Rare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>0.059160</td>\n",
       "      <td>0.059858</td>\n",
       "      <td>-0.459218</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>0.059160</td>\n",
       "      <td>-0.762913</td>\n",
       "      <td>0.434090</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.560975</td>\n",
       "      <td>0.412473</td>\n",
       "      <td>-0.347554</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0.432793</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>0.059160</td>\n",
       "      <td>-0.851067</td>\n",
       "      <td>0.182847</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>-0.474545</td>\n",
       "      <td>-0.473674</td>\n",
       "      <td>-0.560975</td>\n",
       "      <td>1.205859</td>\n",
       "      <td>-0.319638</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 43 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived     SibSp     Parch  relatives  Age_Class  Fare_Per_Person  \\\n",
       "0         0  0.432793 -0.473674   0.059160   0.059858        -0.459218   \n",
       "1         1  0.432793 -0.473674   0.059160  -0.762913         0.434090   \n",
       "2         1 -0.474545 -0.473674  -0.560975   0.412473        -0.347554   \n",
       "3         1  0.432793 -0.473674   0.059160  -0.851067         0.182847   \n",
       "4         0 -0.474545 -0.473674  -0.560975   1.205859        -0.319638   \n",
       "\n",
       "   Pclass_Class1  Pclass_Class2  Pclass_Class3  Sex_female  ...  Deck_D  \\\n",
       "0              0              0              1           0  ...       0   \n",
       "1              1              0              0           1  ...       0   \n",
       "2              0              0              1           1  ...       0   \n",
       "3              1              0              0           1  ...       0   \n",
       "4              0              0              1           0  ...       0   \n",
       "\n",
       "   Deck_E  Deck_F  Deck_G  Deck_U  Title_Master  Title_Miss  Title_Mr  \\\n",
       "0       0       0       0       1             0           0         1   \n",
       "1       0       0       0       0             0           0         0   \n",
       "2       0       0       0       1             0           1         0   \n",
       "3       0       0       0       0             0           0         0   \n",
       "4       0       0       0       1             0           0         1   \n",
       "\n",
       "   Title_Mrs  Title_Rare  \n",
       "0          0           0  \n",
       "1          1           0  \n",
       "2          0           0  \n",
       "3          1           0  \n",
       "4          0           0  \n",
       "\n",
       "[5 rows x 43 columns]"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data_ss.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Test\n",
    "test_encode_col_list = list(test_data.select_dtypes(include=['object']).columns)\n",
    "for i in test_encode_col_list:\n",
    "    test_data_ss = pd.concat([test_data_ss,pd.get_dummies(test_data_ss[i], prefix=i)],axis=1)\n",
    "    test_data_ss.drop(i, axis = 1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(418, 43)"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data_ss.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>relatives</th>\n",
       "      <th>Age_Class</th>\n",
       "      <th>Fare_Per_Person</th>\n",
       "      <th>Pclass_Class1</th>\n",
       "      <th>Pclass_Class2</th>\n",
       "      <th>Pclass_Class3</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>...</th>\n",
       "      <th>Deck_D</th>\n",
       "      <th>Deck_E</th>\n",
       "      <th>Deck_F</th>\n",
       "      <th>Deck_G</th>\n",
       "      <th>Deck_U</th>\n",
       "      <th>Title_Master</th>\n",
       "      <th>Title_Miss</th>\n",
       "      <th>Title_Mr</th>\n",
       "      <th>Title_Mrs</th>\n",
       "      <th>Title_Rare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>-0.499470</td>\n",
       "      <td>-0.400248</td>\n",
       "      <td>-0.553443</td>\n",
       "      <td>0.027878</td>\n",
       "      <td>-0.401204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>0.616992</td>\n",
       "      <td>-0.400248</td>\n",
       "      <td>0.105643</td>\n",
       "      <td>1.228201</td>\n",
       "      <td>-0.513662</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>-0.499470</td>\n",
       "      <td>-0.400248</td>\n",
       "      <td>-0.553443</td>\n",
       "      <td>-0.322216</td>\n",
       "      <td>-0.344975</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>-0.499470</td>\n",
       "      <td>-0.400248</td>\n",
       "      <td>-0.553443</td>\n",
       "      <td>1.003140</td>\n",
       "      <td>-0.373089</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>0.616992</td>\n",
       "      <td>0.619896</td>\n",
       "      <td>0.764728</td>\n",
       "      <td>1.003140</td>\n",
       "      <td>-0.485547</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 43 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId     SibSp     Parch  relatives  Age_Class  Fare_Per_Person  \\\n",
       "0          892 -0.499470 -0.400248  -0.553443   0.027878        -0.401204   \n",
       "1          893  0.616992 -0.400248   0.105643   1.228201        -0.513662   \n",
       "2          894 -0.499470 -0.400248  -0.553443  -0.322216        -0.344975   \n",
       "3          895 -0.499470 -0.400248  -0.553443   1.003140        -0.373089   \n",
       "4          896  0.616992  0.619896   0.764728   1.003140        -0.485547   \n",
       "\n",
       "   Pclass_Class1  Pclass_Class2  Pclass_Class3  Sex_female  ...  Deck_D  \\\n",
       "0              0              0              1           0  ...       0   \n",
       "1              0              0              1           1  ...       0   \n",
       "2              0              1              0           0  ...       0   \n",
       "3              0              0              1           0  ...       0   \n",
       "4              0              0              1           1  ...       0   \n",
       "\n",
       "   Deck_E  Deck_F  Deck_G  Deck_U  Title_Master  Title_Miss  Title_Mr  \\\n",
       "0       0       0       0       1             0           0         1   \n",
       "1       0       0       0       1             0           0         0   \n",
       "2       0       0       0       1             0           0         1   \n",
       "3       0       0       0       1             0           0         1   \n",
       "4       0       0       0       1             0           0         0   \n",
       "\n",
       "   Title_Mrs  Title_Rare  \n",
       "0          0           0  \n",
       "1          1           0  \n",
       "2          0           0  \n",
       "3          0           0  \n",
       "4          1           0  \n",
       "\n",
       "[5 rows x 43 columns]"
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data_ss.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Entrainement du modele "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = train_data_ss.drop(\"Survived\", axis=1)\n",
    "Y_train = train_data_ss[\"Survived\"]\n",
    "X_test  = test_data_ss.drop(\"PassengerId\", axis=1).copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 42)"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891,)"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(418, 42)"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 42 columns):\n",
      "SibSp                  891 non-null float64\n",
      "Parch                  891 non-null float64\n",
      "relatives              891 non-null float64\n",
      "Age_Class              891 non-null float64\n",
      "Fare_Per_Person        891 non-null float64\n",
      "Pclass_Class1          891 non-null uint8\n",
      "Pclass_Class2          891 non-null uint8\n",
      "Pclass_Class3          891 non-null uint8\n",
      "Sex_female             891 non-null uint8\n",
      "Sex_male               891 non-null uint8\n",
      "Age_Adults             891 non-null uint8\n",
      "Age_Children           891 non-null uint8\n",
      "Age_Middle Age         891 non-null uint8\n",
      "Age_Retired            891 non-null uint8\n",
      "Age_Senior             891 non-null uint8\n",
      "Age_Teens              891 non-null uint8\n",
      "Age_Young Adults       891 non-null uint8\n",
      "Age_Youngsters         891 non-null uint8\n",
      "Fare_Extremely High    891 non-null uint8\n",
      "Fare_Extremely Low     891 non-null uint8\n",
      "Fare_High              891 non-null uint8\n",
      "Fare_Low               891 non-null uint8\n",
      "Fare_Very High         891 non-null uint8\n",
      "Fare_Very Low          891 non-null uint8\n",
      "Embarked_C             891 non-null uint8\n",
      "Embarked_Q             891 non-null uint8\n",
      "Embarked_S             891 non-null uint8\n",
      "travelled_alone_No     891 non-null uint8\n",
      "travelled_alone_Yes    891 non-null uint8\n",
      "Deck_A                 891 non-null uint8\n",
      "Deck_B                 891 non-null uint8\n",
      "Deck_C                 891 non-null uint8\n",
      "Deck_D                 891 non-null uint8\n",
      "Deck_E                 891 non-null uint8\n",
      "Deck_F                 891 non-null uint8\n",
      "Deck_G                 891 non-null uint8\n",
      "Deck_U                 891 non-null uint8\n",
      "Title_Master           891 non-null uint8\n",
      "Title_Miss             891 non-null uint8\n",
      "Title_Mr               891 non-null uint8\n",
      "Title_Mrs              891 non-null uint8\n",
      "Title_Rare             891 non-null uint8\n",
      "dtypes: float64(5), uint8(37)\n",
      "memory usage: 67.1 KB\n"
     ]
    }
   ],
   "source": [
    "X_train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Utilisation de RandomForest() comme modèle de decision \n",
    "### D'après nos recherches et les tests effectués ce modèle est le plus précis \n",
    "### Nous avons fait un test avec les modèles (KNeighborsClassifier(),LogisticRegression) avec des resultats moins précis. \n",
    "\n",
    "random_forest = RandomForestClassifier(n_estimators=100)\n",
    "random_forest.fit(X_train, Y_train)\n",
    "\n",
    "random_forest_predictions = random_forest.predict(X_test)\n",
    "\n",
    "rf_data = pd.read_csv('C:\\\\Users/balde/OneDrive/Documents/Cours M2/IPSSI/Cours Python/Projet du 01-07-2020/Projet Titanic/kaggle/input/titanic/test.csv')\n",
    "rf_data.insert((rf_data.shape[1]),'Survived',random_forest_predictions)\n",
    "\n",
    "rf_data.to_csv('My_submission.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

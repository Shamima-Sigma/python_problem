{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('emails.csv')"
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
       "      <th>text</th>\n",
       "      <th>spam</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Subject: top - level logo and business identit...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Subject: your trusted source for prescription ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Subject: rely on us for your online prescripti...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Subject: guzzle like a fountain  spur m rocks ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Subject: are you losing ? the answer would ama...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5707</th>\n",
       "      <td>Subject: re : research and development charges...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5708</th>\n",
       "      <td>Subject: re : receipts from visit  jim ,  than...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5709</th>\n",
       "      <td>Subject: re : enron case study update  wow ! a...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5710</th>\n",
       "      <td>Subject: re : interest  david ,  please , call...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5711</th>\n",
       "      <td>Subject: news : aurora 5 . 2 update  aurora ve...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5712 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   text spam\n",
       "0     Subject: top - level logo and business identit...    1\n",
       "1     Subject: your trusted source for prescription ...    1\n",
       "2     Subject: rely on us for your online prescripti...    1\n",
       "3     Subject: guzzle like a fountain  spur m rocks ...    1\n",
       "4     Subject: are you losing ? the answer would ama...    1\n",
       "...                                                 ...  ...\n",
       "5707  Subject: re : research and development charges...    0\n",
       "5708  Subject: re : receipts from visit  jim ,  than...    0\n",
       "5709  Subject: re : enron case study update  wow ! a...    0\n",
       "5710  Subject: re : interest  david ,  please , call...    0\n",
       "5711  Subject: news : aurora 5 . 2 update  aurora ve...    0\n",
       "\n",
       "[5712 rows x 2 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0                                                                                               4359\n",
       "1                                                                                               1345\n",
       " its termination would not  have such a phenomenal impact on the power situation .  however        1\n",
       " mr suresh prabhu                                                                                  1\n",
       "Name: spam, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['spam'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop_duplicates(inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
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
       "      <th>text</th>\n",
       "      <th>spam</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Subject: top - level logo and business identit...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Subject: your trusted source for prescription ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Subject: rely on us for your online prescripti...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Subject: guzzle like a fountain  spur m rocks ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Subject: are you losing ? the answer would ama...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5707</th>\n",
       "      <td>Subject: re : research and development charges...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5708</th>\n",
       "      <td>Subject: re : receipts from visit  jim ,  than...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5709</th>\n",
       "      <td>Subject: re : enron case study update  wow ! a...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5710</th>\n",
       "      <td>Subject: re : interest  david ,  please , call...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5711</th>\n",
       "      <td>Subject: news : aurora 5 . 2 update  aurora ve...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5676 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   text spam\n",
       "0     Subject: top - level logo and business identit...    1\n",
       "1     Subject: your trusted source for prescription ...    1\n",
       "2     Subject: rely on us for your online prescripti...    1\n",
       "3     Subject: guzzle like a fountain  spur m rocks ...    1\n",
       "4     Subject: are you losing ? the answer would ama...    1\n",
       "...                                                 ...  ...\n",
       "5707  Subject: re : research and development charges...    0\n",
       "5708  Subject: re : receipts from visit  jim ,  than...    0\n",
       "5709  Subject: re : enron case study update  wow ! a...    0\n",
       "5710  Subject: re : interest  david ,  please , call...    0\n",
       "5711  Subject: news : aurora 5 . 2 update  aurora ve...    0\n",
       "\n",
       "[5676 rows x 2 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0                                                                                               4326\n",
       "1                                                                                               1345\n",
       " its termination would not  have such a phenomenal impact on the power situation .  however        1\n",
       " mr suresh prabhu                                                                                  1\n",
       "Name: spam, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['spam'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "text    1\n",
       "spam    3\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropna(how = 'any' , axis = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "text    0\n",
       "spam    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Separate x and y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.text.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df.spam.values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Split dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv = CountVectorizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "xtrain = cv.fit_transform(x_train)"
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
       "array([[ 0,  0,  0, ...,  0,  0,  0],\n",
       "       [ 0,  0,  0, ...,  0,  0,  0],\n",
       "       [ 0,  0,  0, ...,  0,  0,  0],\n",
       "       ...,\n",
       "       [ 0,  0,  0, ...,  0,  0,  0],\n",
       "       [11,  0,  0, ...,  0,  0,  0],\n",
       "       [ 0,  0,  0, ...,  0,  0,  0]], dtype=int64)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xtrain.toarray()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ML algorithm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = MultinomialNB()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(xtrain, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "xtest = cv.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [2, 0, 0, ..., 0, 0, 0],\n",
       "       ...,\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0],\n",
       "       [0, 0, 0, ..., 0, 0, 0]], dtype=int64)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xtest.toarray()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9911894273127754"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.score(xtest, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "emails = ['Hey I am looking for machine learning tutorial.','Hey, you win an iphone x giveaway for free. Please do the the survey.']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv_mails = cv.transform(emails)"
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
       "array(['0', '1'], dtype='<U92')"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.predict(cv_mails)"
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
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

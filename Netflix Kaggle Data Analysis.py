{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np #loading numpy "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd #loading pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "original_file = pd.read_csv('netflix.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
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
       "      <th>show_id</th>\n",
       "      <th>type</th>\n",
       "      <th>title</th>\n",
       "      <th>director</th>\n",
       "      <th>cast</th>\n",
       "      <th>country</th>\n",
       "      <th>date_added</th>\n",
       "      <th>release_year</th>\n",
       "      <th>rating</th>\n",
       "      <th>duration</th>\n",
       "      <th>listed_in</th>\n",
       "      <th>description</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>s1</td>\n",
       "      <td>TV Show</td>\n",
       "      <td>3%</td>\n",
       "      <td>NaN</td>\n",
       "      <td>João Miguel, Bianca Comparato, Michel Gomes, R...</td>\n",
       "      <td>Brazil</td>\n",
       "      <td>August 14, 2020</td>\n",
       "      <td>2020</td>\n",
       "      <td>TV-MA</td>\n",
       "      <td>4 Seasons</td>\n",
       "      <td>International TV Shows, TV Dramas, TV Sci-Fi &amp;...</td>\n",
       "      <td>In a future where the elite inhabit an island ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>s2</td>\n",
       "      <td>Movie</td>\n",
       "      <td>7:19</td>\n",
       "      <td>Jorge Michel Grau</td>\n",
       "      <td>Demián Bichir, Héctor Bonilla, Oscar Serrano, ...</td>\n",
       "      <td>Mexico</td>\n",
       "      <td>December 23, 2016</td>\n",
       "      <td>2016</td>\n",
       "      <td>TV-MA</td>\n",
       "      <td>93 min</td>\n",
       "      <td>Dramas, International Movies</td>\n",
       "      <td>After a devastating earthquake hits Mexico Cit...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>s3</td>\n",
       "      <td>Movie</td>\n",
       "      <td>23:59</td>\n",
       "      <td>Gilbert Chan</td>\n",
       "      <td>Tedd Chan, Stella Chung, Henley Hii, Lawrence ...</td>\n",
       "      <td>Singapore</td>\n",
       "      <td>December 20, 2018</td>\n",
       "      <td>2011</td>\n",
       "      <td>R</td>\n",
       "      <td>78 min</td>\n",
       "      <td>Horror Movies, International Movies</td>\n",
       "      <td>When an army recruit is found dead, his fellow...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>s4</td>\n",
       "      <td>Movie</td>\n",
       "      <td>9</td>\n",
       "      <td>Shane Acker</td>\n",
       "      <td>Elijah Wood, John C. Reilly, Jennifer Connelly...</td>\n",
       "      <td>United States</td>\n",
       "      <td>November 16, 2017</td>\n",
       "      <td>2009</td>\n",
       "      <td>PG-13</td>\n",
       "      <td>80 min</td>\n",
       "      <td>Action &amp; Adventure, Independent Movies, Sci-Fi...</td>\n",
       "      <td>In a postapocalyptic world, rag-doll robots hi...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>s5</td>\n",
       "      <td>Movie</td>\n",
       "      <td>21</td>\n",
       "      <td>Robert Luketic</td>\n",
       "      <td>Jim Sturgess, Kevin Spacey, Kate Bosworth, Aar...</td>\n",
       "      <td>United States</td>\n",
       "      <td>January 1, 2020</td>\n",
       "      <td>2008</td>\n",
       "      <td>PG-13</td>\n",
       "      <td>123 min</td>\n",
       "      <td>Dramas</td>\n",
       "      <td>A brilliant group of students become card-coun...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  show_id     type  title           director  \\\n",
       "0      s1  TV Show     3%                NaN   \n",
       "1      s2    Movie   7:19  Jorge Michel Grau   \n",
       "2      s3    Movie  23:59       Gilbert Chan   \n",
       "3      s4    Movie      9        Shane Acker   \n",
       "4      s5    Movie     21     Robert Luketic   \n",
       "\n",
       "                                                cast        country  \\\n",
       "0  João Miguel, Bianca Comparato, Michel Gomes, R...         Brazil   \n",
       "1  Demián Bichir, Héctor Bonilla, Oscar Serrano, ...         Mexico   \n",
       "2  Tedd Chan, Stella Chung, Henley Hii, Lawrence ...      Singapore   \n",
       "3  Elijah Wood, John C. Reilly, Jennifer Connelly...  United States   \n",
       "4  Jim Sturgess, Kevin Spacey, Kate Bosworth, Aar...  United States   \n",
       "\n",
       "          date_added  release_year rating   duration  \\\n",
       "0    August 14, 2020          2020  TV-MA  4 Seasons   \n",
       "1  December 23, 2016          2016  TV-MA     93 min   \n",
       "2  December 20, 2018          2011      R     78 min   \n",
       "3  November 16, 2017          2009  PG-13     80 min   \n",
       "4    January 1, 2020          2008  PG-13    123 min   \n",
       "\n",
       "                                           listed_in  \\\n",
       "0  International TV Shows, TV Dramas, TV Sci-Fi &...   \n",
       "1                       Dramas, International Movies   \n",
       "2                Horror Movies, International Movies   \n",
       "3  Action & Adventure, Independent Movies, Sci-Fi...   \n",
       "4                                             Dramas   \n",
       "\n",
       "                                         description  \n",
       "0  In a future where the elite inhabit an island ...  \n",
       "1  After a devastating earthquake hits Mexico Cit...  \n",
       "2  When an army recruit is found dead, his fellow...  \n",
       "3  In a postapocalyptic world, rag-doll robots hi...  \n",
       "4  A brilliant group of students become card-coun...  "
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "original_file.head() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_usa = original_file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
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
       "      <th>show_id</th>\n",
       "      <th>type</th>\n",
       "      <th>title</th>\n",
       "      <th>director</th>\n",
       "      <th>cast</th>\n",
       "      <th>country</th>\n",
       "      <th>date_added</th>\n",
       "      <th>release_year</th>\n",
       "      <th>rating</th>\n",
       "      <th>duration</th>\n",
       "      <th>listed_in</th>\n",
       "      <th>description</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>7782</th>\n",
       "      <td>s7783</td>\n",
       "      <td>Movie</td>\n",
       "      <td>Zozo</td>\n",
       "      <td>Josef Fares</td>\n",
       "      <td>Imad Creidi, Antoinette Turk, Elias Gergi, Car...</td>\n",
       "      <td>Sweden, Czech Republic, United Kingdom, Denmar...</td>\n",
       "      <td>October 19, 2020</td>\n",
       "      <td>2005</td>\n",
       "      <td>TV-MA</td>\n",
       "      <td>99 min</td>\n",
       "      <td>Dramas, International Movies</td>\n",
       "      <td>When Lebanon's Civil War deprives Zozo of his ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7783</th>\n",
       "      <td>s7784</td>\n",
       "      <td>Movie</td>\n",
       "      <td>Zubaan</td>\n",
       "      <td>Mozez Singh</td>\n",
       "      <td>Vicky Kaushal, Sarah-Jane Dias, Raaghav Chanan...</td>\n",
       "      <td>India</td>\n",
       "      <td>March 2, 2019</td>\n",
       "      <td>2015</td>\n",
       "      <td>TV-14</td>\n",
       "      <td>111 min</td>\n",
       "      <td>Dramas, International Movies, Music &amp; Musicals</td>\n",
       "      <td>A scrappy but poor boy worms his way into a ty...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7784</th>\n",
       "      <td>s7785</td>\n",
       "      <td>Movie</td>\n",
       "      <td>Zulu Man in Japan</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Nasty C</td>\n",
       "      <td>NaN</td>\n",
       "      <td>September 25, 2020</td>\n",
       "      <td>2019</td>\n",
       "      <td>TV-MA</td>\n",
       "      <td>44 min</td>\n",
       "      <td>Documentaries, International Movies, Music &amp; M...</td>\n",
       "      <td>In this documentary, South African rapper Nast...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7785</th>\n",
       "      <td>s7786</td>\n",
       "      <td>TV Show</td>\n",
       "      <td>Zumbo's Just Desserts</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Adriano Zumbo, Rachel Khoo</td>\n",
       "      <td>Australia</td>\n",
       "      <td>October 31, 2020</td>\n",
       "      <td>2019</td>\n",
       "      <td>TV-PG</td>\n",
       "      <td>1 Season</td>\n",
       "      <td>International TV Shows, Reality TV</td>\n",
       "      <td>Dessert wizard Adriano Zumbo looks for the nex...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7786</th>\n",
       "      <td>s7787</td>\n",
       "      <td>Movie</td>\n",
       "      <td>ZZ TOP: THAT LITTLE OL' BAND FROM TEXAS</td>\n",
       "      <td>Sam Dunn</td>\n",
       "      <td>NaN</td>\n",
       "      <td>United Kingdom, Canada, United States</td>\n",
       "      <td>March 1, 2020</td>\n",
       "      <td>2019</td>\n",
       "      <td>TV-MA</td>\n",
       "      <td>90 min</td>\n",
       "      <td>Documentaries, Music &amp; Musicals</td>\n",
       "      <td>This documentary delves into the mystique behi...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     show_id     type                                    title     director  \\\n",
       "7782   s7783    Movie                                     Zozo  Josef Fares   \n",
       "7783   s7784    Movie                                   Zubaan  Mozez Singh   \n",
       "7784   s7785    Movie                        Zulu Man in Japan          NaN   \n",
       "7785   s7786  TV Show                    Zumbo's Just Desserts          NaN   \n",
       "7786   s7787    Movie  ZZ TOP: THAT LITTLE OL' BAND FROM TEXAS     Sam Dunn   \n",
       "\n",
       "                                                   cast  \\\n",
       "7782  Imad Creidi, Antoinette Turk, Elias Gergi, Car...   \n",
       "7783  Vicky Kaushal, Sarah-Jane Dias, Raaghav Chanan...   \n",
       "7784                                            Nasty C   \n",
       "7785                         Adriano Zumbo, Rachel Khoo   \n",
       "7786                                                NaN   \n",
       "\n",
       "                                                country          date_added  \\\n",
       "7782  Sweden, Czech Republic, United Kingdom, Denmar...    October 19, 2020   \n",
       "7783                                              India       March 2, 2019   \n",
       "7784                                                NaN  September 25, 2020   \n",
       "7785                                          Australia    October 31, 2020   \n",
       "7786              United Kingdom, Canada, United States       March 1, 2020   \n",
       "\n",
       "      release_year rating  duration  \\\n",
       "7782          2005  TV-MA    99 min   \n",
       "7783          2015  TV-14   111 min   \n",
       "7784          2019  TV-MA    44 min   \n",
       "7785          2019  TV-PG  1 Season   \n",
       "7786          2019  TV-MA    90 min   \n",
       "\n",
       "                                              listed_in  \\\n",
       "7782                       Dramas, International Movies   \n",
       "7783     Dramas, International Movies, Music & Musicals   \n",
       "7784  Documentaries, International Movies, Music & M...   \n",
       "7785                 International TV Shows, Reality TV   \n",
       "7786                    Documentaries, Music & Musicals   \n",
       "\n",
       "                                            description  \n",
       "7782  When Lebanon's Civil War deprives Zozo of his ...  \n",
       "7783  A scrappy but poor boy worms his way into a ty...  \n",
       "7784  In this documentary, South African rapper Nast...  \n",
       "7785  Dessert wizard Adriano Zumbo looks for the nex...  \n",
       "7786  This documentary delves into the mystique behi...  "
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_usa.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "show_id         7787\n",
       "type            7787\n",
       "title           7787\n",
       "director        5398\n",
       "cast            7069\n",
       "country         7280\n",
       "date_added      7777\n",
       "release_year    7787\n",
       "rating          7780\n",
       "duration        7787\n",
       "listed_in       7787\n",
       "description     7787\n",
       "dtype: int64"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_usa.count() #counts the number of observations for all columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "TV-MA    2863\n",
       "TV-14    1931\n",
       "TV-PG     806\n",
       "R         665\n",
       "PG-13     386\n",
       "TV-Y      280\n",
       "TV-Y7     271\n",
       "PG        247\n",
       "TV-G      194\n",
       "NR         84\n",
       "Name: rating, dtype: int64"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_usa['rating'].value_counts().head(10) #counts the number of observations for the column 'rating'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "movies_usa = original_file.loc[(original_file['country'] == 'United States') & \n",
    "                               (original_file['type'] == 'Movie')] #filters object by parameters"
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
       "2017    279\n",
       "2018    238\n",
       "2019    210\n",
       "Name: release_year, dtype: int64"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_usa['release_year'].value_counts().head(3) #counts the number of observations for each release year"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3                                Shane Acker\n",
       "4                             Robert Luketic\n",
       "7                             Kevin Reynolds\n",
       "10                              Zak Hilditch\n",
       "14                                John Suits\n",
       "25    Lyric R. Cabral, David Felix Sutcliffe\n",
       "33                          Fernando Lebrija\n",
       "Name: director, dtype: object"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_usa['director'].head(7) #indexes for column director only"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pandas.core.frame.DataFrame"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(movies_usa) # looks at structure of data"
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
       "3        80 min\n",
       "4       123 min\n",
       "7       119 min\n",
       "10      103 min\n",
       "14       91 min\n",
       "         ...   \n",
       "7758    101 min\n",
       "7771     12 min\n",
       "7774    158 min\n",
       "7778     88 min\n",
       "7781     88 min\n",
       "Name: duration, Length: 1850, dtype: object"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_usa.duration #like a SQL query syntax, gives colummn of duration for all rows. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "show_type = original_file[['type','duration','rating']]  #selecting columns type duration and rating"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
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
       "      <th>type</th>\n",
       "      <th>duration</th>\n",
       "      <th>rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>TV Show</td>\n",
       "      <td>4 Seasons</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Movie</td>\n",
       "      <td>93 min</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Movie</td>\n",
       "      <td>78 min</td>\n",
       "      <td>R</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Movie</td>\n",
       "      <td>80 min</td>\n",
       "      <td>PG-13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Movie</td>\n",
       "      <td>123 min</td>\n",
       "      <td>PG-13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>TV Show</td>\n",
       "      <td>1 Season</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Movie</td>\n",
       "      <td>95 min</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Movie</td>\n",
       "      <td>119 min</td>\n",
       "      <td>R</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Movie</td>\n",
       "      <td>118 min</td>\n",
       "      <td>TV-14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Movie</td>\n",
       "      <td>143 min</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      type   duration rating\n",
       "0  TV Show  4 Seasons  TV-MA\n",
       "1    Movie     93 min  TV-MA\n",
       "2    Movie     78 min      R\n",
       "3    Movie     80 min  PG-13\n",
       "4    Movie    123 min  PG-13\n",
       "5  TV Show   1 Season  TV-MA\n",
       "6    Movie     95 min  TV-MA\n",
       "7    Movie    119 min      R\n",
       "8    Movie    118 min  TV-14\n",
       "9    Movie    143 min  TV-MA"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_type.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
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
       "      <th>type</th>\n",
       "      <th>duration</th>\n",
       "      <th>rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>TV Show</td>\n",
       "      <td>4 Seasons</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Movie</td>\n",
       "      <td>93 min</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Movie</td>\n",
       "      <td>78 min</td>\n",
       "      <td>R</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Movie</td>\n",
       "      <td>80 min</td>\n",
       "      <td>PG-13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>TV Show</td>\n",
       "      <td>1 Season</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7782</th>\n",
       "      <td>Movie</td>\n",
       "      <td>99 min</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7783</th>\n",
       "      <td>Movie</td>\n",
       "      <td>111 min</td>\n",
       "      <td>TV-14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7784</th>\n",
       "      <td>Movie</td>\n",
       "      <td>44 min</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7785</th>\n",
       "      <td>TV Show</td>\n",
       "      <td>1 Season</td>\n",
       "      <td>TV-PG</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7786</th>\n",
       "      <td>Movie</td>\n",
       "      <td>90 min</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>7786 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         type   duration rating\n",
       "0     TV Show  4 Seasons  TV-MA\n",
       "1       Movie     93 min  TV-MA\n",
       "2       Movie     78 min      R\n",
       "3       Movie     80 min  PG-13\n",
       "5     TV Show   1 Season  TV-MA\n",
       "...       ...        ...    ...\n",
       "7782    Movie     99 min  TV-MA\n",
       "7783    Movie    111 min  TV-14\n",
       "7784    Movie     44 min  TV-MA\n",
       "7785  TV Show   1 Season  TV-PG\n",
       "7786    Movie     90 min  TV-MA\n",
       "\n",
       "[7786 rows x 3 columns]"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_type.drop(4, axis = 0) # axis = 0, means rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
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
       "      <th>type</th>\n",
       "      <th>rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>TV Show</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Movie</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Movie</td>\n",
       "      <td>R</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Movie</td>\n",
       "      <td>PG-13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Movie</td>\n",
       "      <td>PG-13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7782</th>\n",
       "      <td>Movie</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7783</th>\n",
       "      <td>Movie</td>\n",
       "      <td>TV-14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7784</th>\n",
       "      <td>Movie</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7785</th>\n",
       "      <td>TV Show</td>\n",
       "      <td>TV-PG</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7786</th>\n",
       "      <td>Movie</td>\n",
       "      <td>TV-MA</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>7787 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         type rating\n",
       "0     TV Show  TV-MA\n",
       "1       Movie  TV-MA\n",
       "2       Movie      R\n",
       "3       Movie  PG-13\n",
       "4       Movie  PG-13\n",
       "...       ...    ...\n",
       "7782    Movie  TV-MA\n",
       "7783    Movie  TV-14\n",
       "7784    Movie  TV-MA\n",
       "7785  TV Show  TV-PG\n",
       "7786    Movie  TV-MA\n",
       "\n",
       "[7787 rows x 2 columns]"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "show_type.drop('duration',axis = 1) # axis = 1, means columns"
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
       "(7787, 12)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "original_file.shape #looks at the shape of the dataframe, 7787 rows and 12 colummns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "col_select = original_file[['director','country','release_year','listed_in']] #column select"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "row_location_3 = original_file.loc[3] #selects row location, can either 'character string' or index location"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "row_location_4 = original_file.iloc[4] #selects row location, index location"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "obs_location = original_file.loc[[2,4],['director','release_year']] #selects the row index and column (can be list), can be string or index"
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
       "      <th>director</th>\n",
       "      <th>release_year</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Gilbert Chan</td>\n",
       "      <td>2011</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Robert Luketic</td>\n",
       "      <td>2008</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         director  release_year\n",
       "2    Gilbert Chan          2011\n",
       "4  Robert Luketic          2008"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "obs_location"
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

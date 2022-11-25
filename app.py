
# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import spacy
nlp = spacy.load('pl_core_news_sm')
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import plotly.express as px
import plotly
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud, STOPWORDS


# functions
def make_word_cloud(comment_words, width = 1100, height = 650, colour = "black", colormap = "brg"):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(collocations=False, max_words=250, colormap=colormap, width = width, height = height,
                background_color ='black',
                min_font_size = 14, stopwords = stopwords).generate(comment_words) # , stopwords = stopwords

    fig, ax = plt.subplots(figsize = (width/ 100, height/100), facecolor = colour)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return fig


def prepare_cloud_lexeme_data(data_neutral, data_support, data_attack):

  # neutral df
  neu_text = " ".join(data_neutral['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_neu_text = Counter(neu_text.split(" "))
  df_neu_text = pd.DataFrame( {"word": list(count_dict_df_neu_text.keys()),
                              'neutral #': list(count_dict_df_neu_text.values())} )
  df_neu_text.sort_values(by = 'neutral #', inplace=True, ascending=False)
  df_neu_text.reset_index(inplace=True, drop=True)
  #df_neu_text = df_neu_text[~(df_neu_text.word.isin(stops))]

  # support df
  supp_text = " ".join(data_support['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_supp_text = Counter(supp_text.split(" "))
  df_supp_text = pd.DataFrame( {"word": list(count_dict_df_supp_text.keys()),
                              'support #': list(count_dict_df_supp_text.values())} )

  df_supp_text.sort_values(by = 'support #', inplace=True, ascending=False)
  df_supp_text.reset_index(inplace=True, drop=True)
  #df_supp_text = df_supp_text[~(df_supp_text.word.isin(stops))]

  merg = pd.merge(df_supp_text, df_neu_text, on = 'word', how = 'outer')

  #attack df
  att_text = " ".join(data_attack['clean_Text_lemmatized'].apply(lambda x: " ".join(t for t in set(x.split()))).to_numpy())
  count_dict_df_att_text = Counter(att_text.split(" "))
  df_att_text = pd.DataFrame( {"word": list(count_dict_df_att_text.keys()),
                              'attack #': list(count_dict_df_att_text.values())} )

  df_att_text.sort_values(by = 'attack #', inplace=True, ascending=False)
  df_att_text.reset_index(inplace=True, drop=True)
  #df_att_text = df_att_text[~(df_att_text.word.isin(stops))]

  df2 = pd.merge(merg, df_att_text, on = 'word', how = 'outer')
  df2.fillna(0, inplace=True)
  df2['general #'] = df2['support #'] + df2['attack #'] + df2['neutral #']
  df2['word'] = df2['word'].str.replace("'", "_").replace("”", "_").replace("’", "_")
  return df2



def wordcloud_lexeme(dataframe, lexeme_threshold = 90, analysis_for = 'support', cmap_wordcloud = 'crest'):
  '''
  analysis_for:
  'support',
  'attack',
  'both' (both support and attack)

  cmap_wordcloud: best to choose from:
  gist_heat, flare_r, crest, viridis

  '''
  if analysis_for == 'attack':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'Reds' #gist_heat
    dataframe['% lexeme'] = (round(dataframe['attack #'] / dataframe['general #'], 3) * 100).apply(float) # att
  elif analysis_for == 'both':
    #print(f'Analysis for: {analysis_for} ')
    cmap_wordcloud = 'autumn' #viridis
    dataframe['% lexeme'] = (round((dataframe['support #'] + dataframe['attack #']) / dataframe['general #'], 3) * 100).apply(float) # both supp & att
  else:
    #print(f'Analysis for: {analysis_for} ')
    dataframe['% lexeme'] = (round(dataframe['support #'] / dataframe['general #'], 3) * 100).apply(float) # supp

  dfcloud = dataframe[(dataframe['% lexeme'] >= int(lexeme_threshold)) & (dataframe['general #'] > 2) & (dataframe.word.map(len)>3)]
  #print(f'There are {len(dfcloud)} words for the analysis of language {analysis_for} with % lexeme threshold equal to {lexeme_threshold}.')
  n_words = dfcloud['word'].nunique()
  text = []
  for i in dfcloud.index:
    w = dfcloud.loc[i, 'word']
    w = str(w).strip()
    if analysis_for == 'both':
      n = int(dfcloud.loc[i, 'support #'] + dfcloud.loc[i, 'attack #'])
    else:
      n = int(dfcloud.loc[i, str(analysis_for)+' #']) #  + dfcloud.loc[i, 'attack #']   dfcloud.loc[i, 'support #']+  general
    l = np.repeat(w, n)
    text.extend(l)

  import random
  random.shuffle(text)
  st.write(f"There are {n_words} words.")
  figure_cloud = make_word_cloud(" ".join(text), 1000, 620, '#1E1E1E', str(cmap_wordcloud)) #gist_heat / flare_r crest viridis
  return figure_cloud


def add_spacelines(number=2):
    for i in range(number):
        st.write("\n")


@st.cache(allow_output_mutation=True)
def load_dataset(dataset):
    if dataset == "Testowy korpus":
        df = load_data(r"DebateTVP_June.xlsx")
    return df


def load_data(file_path, indx = True, indx_col = 0):
  '''Parameters:
  file_path: path to your excel or csv file with data,

  indx: boolean - whether there is index column in your file (usually it is the first column) --> default is True

  indx_col: int - if your file has index column, specify column number here --> default is 0 (first column)
  '''
  if indx == True and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".xlsx"):
    data = pd.read_excel(file_path)

  elif indx == True and file_path.endswith(".csv"):
    data = pd.read_csv(file_path, index_col = indx_col)
  elif indx == False and file_path.endswith(".csv"):
    data = pd.read_csv(file_path)
  return data


def lemmatization(dataframe, text_column):
  '''Parameters:
  dataframe: dataframe with your data,

  text_column: name of a column in your dataframe where text is located
  '''
  df = dataframe.copy()
  lemmas = []
  for doc in nlp.pipe(df[text_column].apply(str)):
    lemmas.append([token.lemma_ for token in doc if not (token.is_punct or token.like_num)])
  df[text_column +"_lemmatized"] = lemmas
  return df


def find_emotive_words(dataframe, content_lemmatized_column, uniq_words=False, database = "nawl"):
  '''Parameters:
  dataframe: dataframe with your data,

  content_lemmatized_column: str - name of a column in your dataframe where lemmatized text is located,

  uniq_words: boolean - True if you want to retrieve only unique emotive words from your text data,
  False if you want to retrieve every emotive word (thus, there can be duplicated words),
  --> *by default it is set to False

  database: str - name of an affective database you want to analyse your data with --> type "nawl" or "EMOTION MEANINGS"
  '''
  database = database.upper()
  db_words = "Word"
  db_emotion_category = "Class"

  if database == "NAWL":
    affective_database = pd.read_excel(r"NAWL_full_db.xlsx", index_col=0)
    db_words = "NAWL_word"
    db_emotion_category = "ED_class"
    affective_database = affective_database[affective_database[db_emotion_category] != "N"]
  elif database == "EMOTION MEANINGS":
    affective_database = pd.read_excel(r"uniq_lemma_Emean.xlsx", index_col=0)
    db_words = "lemma"
    db_emotion_category = "classification"
    affective_database = affective_database[affective_database[db_emotion_category] != "NEU" ]
  else:
    affective_database = pd.read_excel(r"joined_scaled_filled_0_NAWL-Sentimenti_db.xlsx", index_col=0)
    affective_database = affective_database[affective_database[db_emotion_category] != "NEU" ]


  affective_database = affective_database[[db_words]]
  affective_database_emotive_words = affective_database[db_words].tolist()

  all_emotive_words = []
  if uniq_words == True:
    for lemmas_list in dataframe[content_lemmatized_column]:
      emotive_words = set(lemmas_list).intersection(affective_database[db_words])
      #emotive_words = " ".join(set(lemmas_list).intersection(affective_database[db_words]))
      all_emotive_words.append(emotive_words)

  elif uniq_words == False:
    for lemmas_list in dataframe[content_lemmatized_column]:
      emotive_words = [w for w in lemmas_list if w in affective_database_emotive_words]
      #emotive_words = " ".join([w for w in lemmas_list if w in affective_database_emotive_words])
      all_emotive_words.append(emotive_words)
  dataframe["Emotive_words"] = all_emotive_words
  return dataframe



def average(dataframe, emotive_words_column, database = "nawl"):
  '''Parameters:
  dataframe: dataframe with your data,

  emotive_words_column: str - name of a column in your dataframe where emotive words are listed,

  database: str - name of an affective database you want to analyse your data with --> type "nawl" or "EMOTION MEANINGS"
  '''
  import warnings
  np.seterr(divide='ignore')
  warnings.filterwarnings(action='ignore', message='Mean of empty slice')

  database = database.upper()

  if database == "EMOTION MEANINGS":
    Emean_db = pd.read_excel(r"uniq_lemma_Emean.xlsx", index_col=0)
    Emean_db = Emean_db[Emean_db["classification"] != "NEU" ]
    emean_emotion_values = ['HAP M', 'ANG M', 'SAD M', 'FEA M', 'DIS M', 'VAL M', 'ARO M', 'SUR M', 'TRU M', 'ANT M']
    emean_words = "lemma"
    emean_cols = [emean_words] + emean_emotion_values    
    affective_database = Emean_db[emean_cols]
    affective_database.set_index(emean_words, inplace=True)

    happ_all = []
    ang_all = []
    sad_all = []
    fea_all = []
    dis_all = []
    val_all = []
    aro_all = []
    sur_all = []
    tru_all = []
    ant_all = []

    happ_all_vals = []
    ang_all_vals = []
    sad_all_vals = []
    fea_all_vals = []
    dis_all_vals = []
    val_all_vals = []
    aro_all_vals = []
    sur_all_vals = []
    tru_all_vals = []
    ant_all_vals = []

    for emotive_words in dataframe[emotive_words_column]:
      individual_scores = []
      values_scores = []
      for emotion_value in emean_emotion_values:
        individual = affective_database.loc[emotive_words][emotion_value].to_numpy(dtype=np.float32).flatten()
        individual_scores.append(individual)

        average = round(np.nanmean(np.array(individual)), 5)
        values_scores.append(average)

      happ_ind = individual_scores[0]
      happ_all.append(list(happ_ind))
      ang_ind = individual_scores[1]
      ang_all.append(list(ang_ind))
      sad_ind = individual_scores[2]
      sad_all.append(list(sad_ind))
      fea_ind = individual_scores[3]
      fea_all.append(list(fea_ind))
      dis_ind = individual_scores[4]
      dis_all.append(list(dis_ind))
      val_ind = individual_scores[5]
      val_all.append(list(val_ind))
      aro_ind = individual_scores[6]
      aro_all.append(list(aro_ind))
      sur_ind = individual_scores[7]
      sur_all.append(list(sur_ind))
      tru_ind = individual_scores[8]
      tru_all.append(list(tru_ind))
      ant_ind = individual_scores[9]
      ant_all.append(list(ant_ind))

      happ_val = values_scores[0]
      happ_all_vals.append(happ_val)
      ang_val = values_scores[1]
      ang_all_vals.append(ang_val)
      sad_val = values_scores[2]
      sad_all_vals.append(sad_val)
      fea_val = values_scores[3]
      fea_all_vals.append(fea_val)
      dis_val = values_scores[4]
      dis_all_vals.append(dis_val)
      val_val = values_scores[5]
      val_all_vals.append(val_val)
      aro_val = values_scores[6]
      aro_all_vals.append(aro_val)
      sur_val = values_scores[7]
      sur_all_vals.append(sur_val)
      tru_val = values_scores[8]
      tru_all_vals.append(tru_val)
      ant_val = values_scores[9]
      ant_all_vals.append(ant_val)

    dataframe["Happiness"] = happ_all_vals
    dataframe["Anger"] = ang_all_vals
    dataframe["Sadness"] = sad_all_vals
    dataframe["Fear"] = fea_all_vals
    dataframe["Disgust"] = dis_all_vals
    dataframe["Valence"] = val_all_vals
    dataframe["Arousal"] = aro_all_vals
    dataframe["Surprise"] = sur_all_vals
    dataframe["Trust"] = tru_all_vals
    dataframe["Anticipation"] = ant_all_vals
  
  else:
    if database == "NAWL":
      NAWL_db = pd.read_excel(r"NAWL_full_db.xlsx", index_col=0)
      NAWL_db = NAWL_db[NAWL_db["ED_class"] != "N"]  
      emotion_values = ['hap_M_all', 'ang_M_all', 'sad_M_all', 'fea_M_all', 'dis_M_all', 'val_M_all', 'aro_M_all']
      nawl_words = "NAWL_word"
      nawl_cols = [nawl_words] + nawl_emotion_values        
      affective_database = NAWL_db[nawl_cols]
      affective_database.set_index(nawl_words, inplace=True)
    else:
      db_emotion_category = "Class"
      db_words = 'Word'
      affective_database = load_data(r"joined_scaled_filled_0_NAWL-Sentimenti_db.xlsx")
      affective_database = affective_database[affective_database[db_emotion_category] != "NEU" ]
      emotion_values = ['Happiness', 'Anger', 'Sadness', 'Fear', 'Disgust', 'Valence', 'Arousal']
      used_cols = [db_words] + emotion_values
      affective_database = affective_database[used_cols]
      affective_database.set_index(db_words, inplace=True)
    
    happ_all = []
    ang_all = []
    sad_all = []
    fea_all = []
    dis_all = []
    val_all = []
    aro_all = []

    happ_all_vals = []
    ang_all_vals = []
    sad_all_vals = []
    fea_all_vals = []
    dis_all_vals = []
    val_all_vals = []
    aro_all_vals = []

    for emotive_words in dataframe[emotive_words_column]:
      individual_scores = []
      values_scores = []
      for emotion_value in emotion_values:
        individual = affective_database.loc[emotive_words][emotion_value].to_numpy(dtype=np.float32).flatten()
        individual_scores.append(individual)

        average = round(np.nanmean(np.array(individual)), 3)
        values_scores.append(average)

      happ_ind = individual_scores[0]
      happ_all.append(list(happ_ind))
      ang_ind = individual_scores[1]
      ang_all.append(list(ang_ind))
      sad_ind = individual_scores[2]
      sad_all.append(list(sad_ind))
      fea_ind = individual_scores[3]
      fea_all.append(list(fea_ind))
      dis_ind = individual_scores[4]
      dis_all.append(list(dis_ind))
      val_ind = individual_scores[5]
      val_all.append(list(val_ind))
      aro_ind = individual_scores[6]
      aro_all.append(list(aro_ind))

      happ_val = values_scores[0]
      happ_all_vals.append(happ_val)
      ang_val = values_scores[1]
      ang_all_vals.append(ang_val)
      sad_val = values_scores[2]
      sad_all_vals.append(sad_val)
      fea_val = values_scores[3]
      fea_all_vals.append(fea_val)
      dis_val = values_scores[4]
      dis_all_vals.append(dis_val)
      val_val = values_scores[5]
      val_all_vals.append(val_val)
      aro_val = values_scores[6]
      aro_all_vals.append(aro_val)

    dataframe["Happiness"] = happ_all_vals
    
    dataframe["Anger"] = ang_all_vals
    dataframe["Sadness"] = sad_all_vals
    dataframe["Fear"] = fea_all_vals
    dataframe["Disgust"] = dis_all_vals
    dataframe["Valence"] = val_all_vals
    dataframe["Arousal"] = aro_all_vals
  return dataframe


def emotion_category(dataframe, emotive_words_column, database = "nawl"):
  '''Parameters:
  dataframe: dataframe with your data,

  emotive_words_column: str - name of a column in your dataframe where emotive words are located,

  database: str - name of an affective database you want to analyse your data with --> type "nawl" or "EMOTION MEANINGS"
  '''
  import numpy as np

  database = database.upper()
  db_words = "Word"
  db_emotion_category = "Class"

  if database == "NAWL":
    affective_database = pd.read_excel(r"NAWL_full_db.xlsx", index_col=0)
    db_words = "NAWL_word"
    db_emotion_category = "ED_class"
    affective_database = affective_database[affective_database[db_emotion_category] != "N"]

  elif database == "EMOTION MEANINGS":
    affective_database = pd.read_excel(r"uniq_lemma_Emean.xlsx", index_col=0)
    db_words = "lemma"
    db_emotion_category = "classification"
    affective_database = affective_database[affective_database[db_emotion_category] != "NEU" ]
  else:
    affective_database = load_data(r"emotion_6-categories_NAWL_Sentimenti_db.xlsx")
    affective_database = affective_database[affective_database[db_emotion_category] != "NEU" ]

  affective_database = affective_database[[db_words, db_emotion_category]]
  affective_database.set_index(db_words, inplace=True)
  set_of_words = set(affective_database.index)

  all_emotion_categories = []
  for emotive_words in dataframe[emotive_words_column]:
    emotion_categories = [affective_database[db_emotion_category].loc[str(word)] if str(word) in set_of_words else 'unclassified' for word in emotive_words]
    all_emotion_categories.append(emotion_categories)
  dataframe["Emotion_categories"] = all_emotion_categories
  return dataframe


def count_categories(dataframe, emotion_categories_column, database = "nawl"):
  '''Parameters:
  dataframe: dataframe with data,

  emotion_categories_column: str - name of a column in your dataframe where emotion categories are located,

  database: str - name of an affective database you want to analyse your data with --> type "nawl" or "EMOTION MEANINGS"
  '''
  database = database.upper()
  db_words = "Word"
  db_emotion_category = "Class"

  if database == "NAWL":
    affective_database = pd.read_excel(r"NAWL_full_db.xlsx", index_col=0)
    db_emotion_category = "ED_class"
    affective_database = affective_database[affective_database[db_emotion_category] != "N"]
  elif database == "EMOTION MEANINGS":
    affective_database = pd.read_excel(r"uniq_lemma_Emean.xlsx", index_col=0)
    db_emotion_category = "classification"
    affective_database = affective_database[affective_database[db_emotion_category] != "NEU" ]
  else:
    affective_database = load_data(r"emotion_6-categories_NAWL_Sentimenti_db.xlsx")
    affective_database = affective_database[affective_database[db_emotion_category] != "NEU" ]

  all_categories = affective_database[db_emotion_category].unique().tolist()

  dataframe["merge_indx"] = range(0, len(dataframe))
  from collections import Counter
  dataframe = pd.merge(dataframe, pd.DataFrame([Counter(x) for x in dataframe[emotion_categories_column]]).fillna(0).astype(int).add_prefix("CATEGORY_"), how='left', left_on="merge_indx", right_index=True)
  dataframe.drop(["merge_indx"], axis=1, inplace=True)

  for category in all_categories:
    if not "CATEGORY_"+category in dataframe.columns:
      dataframe["CATEGORY_"+category] = 0
  return dataframe


import time
# page config
st.set_page_config(page_title="Analiza emocji", layout="wide") # centered wide

#####################  page content  #####################3
st.title("Analiza emocji w tekście")
add_spacelines(3)

st.write("#### Metody i narzędzia")
with st.expander("Metoda słownikowa"):
    st.write("""
    Emotion Meanings:

    Wierzba, M., Riegel, M., Kocoń, J., Miłkowski, P., Janz, A., Klessa, K., Juszczyk, K., Konat, B.,
    Grimling, D., Piasecki, M., et al. (2021). Emotion norms for 6000 polish word meanings with a
    direct mapping to the polish wordnet. Behavior Research Methods, pages 1–16.
    https://link.springer.com/article/10.3758/s13428-021-01697-0

    \n

    NAWL:

    Wierzba, M., Riegel, M., Wypych, M., Jednoróg, K., Turnau, P., Grabowska, A., and Marchewka, A.
    (2015). Basic Emotions in the Nencki Affective Word List (NAWL BE): New Method of Classifying
    Emotional Stimuli. PLOS ONE, 10(7):e0132305. \n
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0132305

    https://exp.lobi.nencki.gov.pl/nawl-analysis
    
    \n
    
    EMEAN-NAWL:
    
    Połączony słownik Emotion Meanings i NAWL na wspólnych wymiarach 5 emocji podstawowych, walencji i pobudzenia. 
    Wyniki analizy opracowano na podstawie skali znormalizowanej.
    """)

add_spacelines(1)

with st.expander("Metoda z użyciem deep learningu"):
    st.write("""
    **eevvgg/PaReS-sentimenTw-political-PL**

    Model bazowany na architekturze BERT'a wytrenowany do rozpoznawania 3 kategorii sentymentu:
    https://huggingface.co/eevvgg/PaReS-sentimenTw-political-PL

    BERT:
    Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).
    Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

    \n

    **cardiffnlp/xlm-twitter-politics-sentiment**

    https://huggingface.co/cardiffnlp/xlm-twitter-politics-sentiment
    """)

add_spacelines(3)

#from transformers import pipeline
#model_path = "eevvgg/PaReS-sentimenTw-political-PL"
#model_path = "cardiffnlp/xlm-twitter-politics-sentiment"


#  *********************** sidebar  *********************
with st.sidebar:
    #standard
    st.title("Parametry Analizy")
    st.write("**Wybierz korpus**")
    if ('test_korpus' and "text_input") not in st.session_state:
        st.session_state['test_korpus'] = False
        st.session_state['text_input'] = False
    if 'placeholder' not in st.session_state:
        st.session_state['placeholder'] = "Tak wysokiego poziomu hipokryzji w polityce nie było chyba nigdy w III RP!"

    box_testowy = st.checkbox("Testowy korpus", value=False,
                            disabled=st.session_state.text_input, key="test_korpus")
    box_txt_input = st.checkbox("Własny tekst", value=False,
                            disabled=st.session_state.test_korpus, key = "text_input")

    if box_testowy:
        data = load_dataset("Testowy korpus")
    elif box_txt_input:
        txt_input = st.text_area(label="Wprowadź tekst", placeholder = st.session_state.placeholder, height = 20)
        if not txt_input:
            st.error('Wprowadź tekst do analizy')
            assert_txt = st.button("Zatwierdź")
            st.stop()
        if len(str(txt_input).split("\n")) > 1:
            txt_list = str(txt_input).split("\n")
            txt_list = [str(t).strip() for t in txt_list]
        else:
            txt_list = [txt_input]
        data = pd.DataFrame({'argument': txt_list})
    add_spacelines(1)
    contents_radio = st.radio("Wybierz analizę", ("Analiza podstawowa", "Analiza rozszerzona"))# ("Metoda słownikowa", "Deep learning model")
    #add_spacelines(1)
    
    if contents_radio == "Analiza podstawowa":        
        contents_radio3 = st.radio("Wybierz język dla tekstu", ("PL", "EN"))
        if contents_radio3 == "PL":
            contents_radio2 = st.radio("Wybierz leksykon", ("EMOTION MEANINGS", "NAWL", "EMEAN-NAWL"))

    #elif contents_radio == "Deep learning model":
        #from transformers import pipeline
        #contents_radio_bert_deep = st.radio("Wybierz model", ("eevvgg/PaReS-sentimenTw-political-PL",
                                                                #"cardiffnlp/xlm-twitter-politics-sentiment", "PaREMO"))
    add_spacelines(1)

    st.write("**Kliknij by zacząć analizę**")
    analise_txt = st.button("Analizuj")

    #alternative
    #form = st.form("my_form")
    #form.write("**Wybierz korpus**")
    #box_testowy = form.checkbox("Testowy korpus", value=False)
    #box_txt_input = form.checkbox("Wprowadź tekst", value=False)
    #txt_input = form.txt_input("Twój tekst", "Oczywiście ze Pan Prezydent \nto nasza duma narodowa!!")
    #if box_testowy:
        #data = load_dataset("Testowy korpus")
    #elif box_txt_input:
        #form.write('\n\n')
        #txt_list = [txt_input]
        #data = pd.DataFrame({'argument': txt_list})
    #add_spacelines(2)
    #form.write('\n\n\n\n\n')
    #contents_radio = form.radio("**Wybierz analizę**", ("Analiza podstawowa", "Analiza szczegółowa"))
    #form.write('\n\n\n\n\n')
    #add_spacelines(2)
    #contents_radio2 = form.radio("**Wybierz leksykon**", ("EMOTION MEANINGS", "NAWL", "EMEAN-NAWL"))
    #form.write('\n\n\n\n\n\n\n')
    #form.write('\n\n\n\n\n')
    #button_analise = st.form_submit_button("Analizuj")



#####################  page content  #####################
if (box_testowy or box_txt_input) and analise_txt and contents_radio == "Analiza podstawowa":
    #if contents_radio3 == "PL":
        #wybrany_leks = contents_radio2
    #else:
        #wybrany_leks = "NRC EmoLex"
    my_data = data.copy()
    wybrany_leks = contents_radio2
    if box_testowy:
        my_data = my_data.sample(n=50)
    my_data = my_data.reset_index(drop=True)
    st.write("#### Analiza w toku ...")

    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    my_data = lemmatization(my_data, "argument")
    my_data = find_emotive_words(my_data, content_lemmatized_column = "argument_lemmatized", database = wybrany_leks)
    my_data = emotion_category(my_data, emotive_words_column= "Emotive_words", database = wybrany_leks)
    my_data = average(my_data, emotive_words_column = "Emotive_words", database = wybrany_leks) 
    my_data = count_categories(my_data, "Emotion_categories", database = wybrany_leks)

    my_data[["Happiness", "Anger", "Sadness", "Fear", "Disgust", "Valence", "Arousal"]] = my_data[["Happiness", "Anger", "Sadness", "Fear", "Disgust", "Valence", "Arousal"]].apply(lambda x: round(x, 3))
    if "Surprise" in my_data.columns:
        my_data[["Happiness", "Anger", "Sadness", "Fear", "Disgust", "Valence", "Arousal", "Surprise", "Anticipation", "Trust"]] = my_data[["Happiness", "Anger", "Sadness", "Fear", "Disgust", "Valence", "Arousal", "Surprise", "Anticipation", "Trust"]].apply(lambda x: round(x, 3))
    add_spacelines(2)
    st.write("#### Wynik analizy")
    st.write(f"Wybrany leksykon: {wybrany_leks}.")
    if wybrany_leks == 'EMOTION MEANINGS':
        num_em = '8'
        author_em = "Roberta Plutchik'a"
    else:
        num_em = "5"
        author_em = "Paul'a Ekmana"
    st.write(f"Dokonano analizy tekstu na wymiarze {num_em} emocji podstawowych według modelu {author_em}.")
    if "Unnamed: 0" in my_data.columns:
        my_data = my_data.drop(["argument_lemmatized", "Unnamed: 0"], axis=1)
    else:
        my_data = my_data.drop(["argument_lemmatized"], axis=1)
    st.dataframe(my_data)

    add_spacelines(2)

    st.write("#### Pobierz wynik analizy")
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(my_data)
    st.download_button(
        label="Kliknij by pobrać CSV",
        data=csv,
        file_name=f'wynik_analiza_emocji-{wybrany_leks}.csv',
        mime='text/csv',
        )

elif (box_testowy or box_txt_input) and analise_txt and contents_radio == "Deep learning model":
    #sentiment_task = pipeline(task = "sentiment-analysis", model = model_path, tokenizer = model_path)
    my_data = data.copy()
    if box_testowy:
        my_data = my_data.sample(n=100)
    my_data = my_data.reset_index(drop=True)
    my_data = my_data[my_data.argument.str.split().map(len) > 1]
    st.write("#### Analiza w toku ...")

    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    
    sequence = my_data.argument.apply(str).values
    if len(sequence) > 500:
        sequence = sequence[:500]

    #modelPaREMO = keras.models.load_model(r"PaREMO_model_original.h5")
    #x_df = laser.embed_sentences(sequence, lang='pl')
    #prediction = modelPaREMO.predict(x = x_df, batch_size=128, verbose=0)
    #labels = np.argmax(prediction, axis=1)
    
    #my_data['emocja-PaREMO'] = labels
    #map_emo = {0: 'neutral',1: 'anger',2: 'disgust',3: 'fear',4: 'joy',5: 'sadness',6: 'neutral'}
    #my_data['emocja-PaREMO'] = my_data['emocja-PaREMO'].map(map_emo)
    
    #sequence = my_data.argument.to_list()
    #sequence = [str(s) if len(str(s)) < 400 else str(s)[:400] for s in sequence]
    #result = sentiment_task(sequence)
    #labels = [i['label'] for i in result]
    #my_data['sentiment-BERT'] = labels
    #my_data['sentiment-BERT'] = my_data['sentiment-BERT'].map({'Neutral':'neutralny','Positive':'pozytywny', 'Negative':'negatywny'})

    st.dataframe(my_data)
    add_spacelines(2)
    st.write("#### Pobierz wynik analizy")
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(my_data)
    st.download_button(
        label="Kliknij by pobrać CSV",
        data=csv,
        file_name=f'wynik_analiza_emocji_PaRes-deep.csv',
        mime='text/csv',
        )

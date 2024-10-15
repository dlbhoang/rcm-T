import pandas as pd
import nltk

# Tải bộ dữ liệu cần thiết
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')  # Để sử dụng word_tokenize
nltk.download('wordnet')  # Để sử dụng lemmatization
nltk.download('stopwords')  # Để sử dụng stopwords
nltk.download('averaged_perceptron_tagger_eng')
# Tạo DataFrame thử nghiệm với một số nội dung
data = {
    "Content": [
        "Tôi rất thích món ăn này! Nó thật sự tuyệt vời. http://example.com",
        "Đồ ăn này tệ quá! Chờ đợi quá lâu và dịch vụ không tốt.",
        "Sản phẩm chất lượng, nhanh chóng và hiệu quả! Thích lắm.",
        "Cảm giác rất đau lòng khi thấy sản phẩm này không như mong đợi. http://test.com",
        "Một trải nghiệm tồi tệ! Không thể chấp nhận.",
        ""
    ]
}

df_test = pd.DataFrame(data)

# In DataFrame ban đầu
print("DataFrame ban đầu:")
print(df_test)

# Định nghĩa hàm simple_text_clean như đã cung cấp
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

def simple_text_clean(dataframe):
    stop_words = set(stopwords.words("english"))

    # Remove HTTP links
    dataframe["Content"] = dataframe["Content"].replace(
        r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
        "", 
        regex=True,
    )
    
    # Remove end of line characters
    dataframe["Content"] = dataframe["Content"].replace(r"[\r\n]+", " ", regex=True)
    
    # Remove numbers, only keep letters
    dataframe["Content"] = dataframe["Content"].replace("[\w]*\d+[\w]*", "", regex=True)
    
    # Remove punctuation
    dataframe["Content"] = dataframe["Content"].replace("[^\w\s]", " ", regex=True)
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for char in punctuation:
        dataframe["Content"] = dataframe["Content"].replace(char, " ")
        
    # Remove multiple spaces with one space
    dataframe["Content"] = dataframe["Content"].replace("[\s]{2,}", " ", regex=True)
    
    # Some lines start with a space, remove them
    dataframe["Content"] = dataframe["Content"].replace("^[\s]{1,}", "", regex=True)
    
    # Some lines end with a space, remove them
    dataframe["Content"] = dataframe["Content"].replace("[\s]{1,}$", "", regex=True)
    
    # Convert to lower case
    dataframe["Content"] = dataframe["Content"].str.lower()
    
    # Remove rows that are empty
    dataframe = dataframe[dataframe["Content"].str.len() > 0]

    # Remove stop words
    def remove_stopwords(text):
        text_split = text.split()
        text = [word for word in text_split if word not in stop_words]
        return " ".join(text)

    dataframe["Content"] = dataframe["Content"].apply(remove_stopwords)

    # Word Net Lemmatizer instead of Stemming, to have a better result
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize_text(text):
        lemmatized = []
        post_tag_list = pos_tag(word_tokenize(text))
        for word, post_tag_val in post_tag_list:
            lemmatized.append(lemmatizer.lemmatize(word, get_wordnet_pos(post_tag_val)))
        text = " ".join(x for x in lemmatized)
        return text

    dataframe["Content"] = dataframe["Content"].apply(lemmatize_text)

    return dataframe

# Áp dụng hàm làm sạch
df_cleaned = simple_text_clean(df_test)

# In DataFrame sau khi làm sạch
print("\nDataFrame sau khi làm sạch:")
print(df_cleaned)

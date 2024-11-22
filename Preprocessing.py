import seaborn as sns

#Definisikan pallet warna untuk tiap distribusi
colors = ["#FF5733", "#33FFC4", "338AFF", "#C833FF", "#FF33E6", "#33FF57"]

#Buat Plot menggunakan pallet warna yang sudah di input
sns.countplot(x = 'label', data = df, palette = color, hue = 'label')
plt.title('class Distribution')
plt.legend(loc = 'upper right')  #legenda di tempatkan di kanan atas
plt.show()

#Bancing data

#Pada columns 1-5
MIN_SIZE = np.amin(df.label.value.counts())
#Mengacak Dataset
acak_df = df.sample(frac=1, random_state=9)

#Memisahkan data yang akan di downsample dari dataset
joy = df[df['label'] == 'joy']
sadness = df[df['label'] == 'sadness']
anger = df[df['label'] == 'anger']
fear = df[df['label'] == 'fear']
love = df[df['label'] == 'love']
suprise = df[df['label'] == 'suprise']

#Memilih secara random data yang telah di pisah
joy_under = joy.sample(MIN_SIZE)
sadness_under = sadness_under.sample(MIN_SIZE)
anger_under = anger.sample(MIN_SIZE)
fear_under = fear.sample(MIN_SIZE)
love_under = love.sample(MIN_SIZE)

df_under = pd.concat([joy_under, sadness_under, anger_under, fear_under, love_under, suprise])

#Mengacak posisi baris
df_shuffled = df_under.sample(frac = 1, random_state = 42).reset_index (drop=True)

#Reset index setelah pengacakan
df_shuffled.reset_index(drop = True, inplace = True)


#Merubah colomns label menjadi numerik
df_shuffled['label_num'] = df_shuffled['label'].map({
	'joy' : 0,
	'sadness' : 1,
	'anger' : 2,
	'fear' : 3,
	'love' : 4, 
	'suprise' : 5
	})

#Pembersihan kalimat kalimat dalam  colomn sentence
import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
	#konversi text menjadi huruf kecil
	text = text.lower()

	doc = nlp(text)
	fitered_tokens = []
	for token in doc:
		if token.is_stop or token.is_punct:
			continue
		else:
			fitered_token.append(token.lemma_)
	return " ".join(fitered_tokens)


#Pembuatan Token
df_shuffled['processed_text'] = df_shuffled["sentence"].apply(preprocess)

#Pemisahan Dataset
x_Train, x_test, y_Train, y_test = train_test_split(
	df_shuffled.processed_text,
	df_shuffled.label_num,
	test_size=0.2,
	random_state=42
	)

from joblib import load
import re
import nltk
from nltk.stem.snowball import SnowballStemmer


path  = 'C:\\Users\\boulo\\Documents\\ANTOINE\\PROJET IA\\Natural_language_processing\\'
path1 = 'C:\\Users\\boulo\\Documents\\ANTOINE\\PROJET IA\\'

TRESHOLD         = .2
logistic_model   = load(path + 'models/logistic_model.joblib'     )
tfidf_vectorizer = load(path + 'tokenizer/tfidf_vectorizer.joblib')
stemmer          = SnowballStemmer('english'               )

abbr = {
	'ninstagram'        : 'instagram',
	'instagramgram'     : 'instagram',
	'ig'                : 'instagram',
	'strainstagramht'   : 'instagram',
	'insta'             : 'instagram',
	'rinstagramht'      : 'instagram',
	'ninstagramguh'     : 'instagram',
	'instagramz'        : 'instagram',
	'sinstagramn'       : 'instagram',
	'binstagramgest'    : 'instagram',
	'pinstagram'        : 'instagram',
	'linstagramht'      : 'instagram',
	'ninstagramg'       : 'instagram',
	'instagramh'        : 'instagram',
	'instagramnor'      : 'instagram',
	'ninstagramht'      : 'instagram',
	'ninstagramgramga'  : 'instagram',
	'finstagramht'      : 'instagram',
	'binstagram'        : 'instagram',
	'hinstagramh'       : 'instagram',
	'ninstagramga'      : 'instagram',
	'toninstagramht'    : 'instagram',
	'minstagramht'      : 'instagram',
	'minstagramt'       : 'instagram',
	
	'nigger'			: 'nigga',
	'niggah'			: 'nigga',
	'nigguh'			: 'nigga',
	'niccuh'			: 'nigga',
	'nicca'				: 'nigga',
	'nig'				: 'nigga',
	
	
	'dwn'               : 'down',
	'dawn'              : 'down',

	'ta'                : 'that',
	'dat'               : 'that',

	'yank'				: 'yankee',
	'dawg'              : 'dude',
	'smh'               : 'head',
	'fr'                : 'real',
	'plz'               : 'please',
	'tf'                : 'wtf',
	'theyr'             : 'are',
	'bc'                : 'because',
	'af'                : 'lot',
	'u'                 : 'you',
	'ppl'               : 'people',
	'dm'                : 'message',
	'bf'                : 'friend',
	'gt'                : 'getting',
	'ya'                : 'yes',
	'na'                : 'no',
	'ur'                : 'your',
	'tryna'             : 'to',
	'lmfao'             : 'lmao',
	'ive'               : 'have'
}

def reduce_repetition(s):
	# use regular expression to find repeated substrings
	pattern = re.compile(r'(.+?)\1{%d,}' % 2)
	match = pattern.search(s)

	# reduce repetition to two occurrences
	while match:
		repeated_substring = match.group(1)
		s = s.replace(match.group(), repeated_substring, 1)
		match = pattern.search(s)

	return s

def remove_points(line):
	l = list(line)

	for i in range(len(l)):
		# print(i, "\t", l[i], "\t", l[i+2])
		if i < len(l) - 2 and l[i] == '.' and (l[i+2] < 'A' or l[i+2] > 'Z'):
			l[i] = ' '

	line = ''.join(l)
	return line

def preprocess(sentence, get='stems'):
	sent = re.sub(r'@([a-zA-Z0-9_]+)'	, 'username', sentence).replace('username:', '') # replace first username
	sent = re.sub(r'http?://\S+'		, 'weblink'	, sent	  )
	sent = re.sub(r'&amp'				, '&'		, sent	  )
	sent = re.sub(r"&#\d+"				, ''		, sent	  )
	
	if sent and sent[0] == '.':
		sent = sent[1:]

	sent = remove_points(sent)
	
	sent = sent   		.replace('RT', '').replace('!', ' ').replace('"', '').replace("\n", ' ')\
						.replace(';', ' ').replace('-', ' ').replace(' and ', ' & ').replace('\'', '')\
						.replace('?', '.').replace(',', '').replace('~', ' ').replace('|', ' ').replace('°', ' ')\
						.replace('`', ' ').replace('~', ' ').replace('*', ' ').replace('+', ' ').replace('/', ' ')\
						.replace(' # ', ' ').replace('http', ' ').replace('t.co', ' ').replace('\\', ' ').replace('&#', ' ')
	for _ in range(4):
		sent = sent.replace('  ', ' ').replace('..', '.').replace(' .', '.') # remove multiple points & space
	
	if sent and sent[0] == ' ':
		sent = sent[1:]

	sent = sent.lower()

	for old in abbr:
		new = abbr[old]
		sent = sent.replace( ' ' + old + ' ', ' ' + new + ' ' ) # add some space arround the world to avoid matching a part of a word

	sent = reduce_repetition(sent)

	if get == 'sentence':
		return sent
	
	tokens = nltk.word_tokenize(sent)

	for i in range(len(tokens) - 1, 0, -1):
		if len(tokens[i].strip()) == 0 or (len(tokens[i].strip()) == 1 and tokens[i].strip() != 'a' and tokens[i].strip() != 'i' and tokens[i].strip() != '&'):
			tokens.pop(i)

	if get == 'tokens':
		return tokens

	stems = []

	for tok in tokens:
		stems.append(stemmer.stem(tok))

	for i in range(len(stems) - 1, 0, -1):
		if len(stems[i].strip()) == 0:
			stems.pop(i)
		elif len(stems[i].strip()) == 1:
			if 'a' < stems[i].strip() <= 'z'and stems[i].strip() != 'i':
				stems.pop(i)


	for i in range(len(stems) - 1, 0, -1):
		arr = stems[i].split('.')
		stems[i] = arr[0]
		for j in range(1, len(arr)):
			stems.insert(i+j, arr[j])

	for i in range(len(stems)):
		if stems[i] in abbr.keys():
			stems[i] = abbr[stems[i]]

	return stems

# use by running this script
def predict_hate(sentences):

	if type(sentences) is list:

		X_sentences = [" ".join(preprocess(i)) for i in sentences]
		X_sentences = tfidf_vectorizer.transform(X_sentences).toarray()
		predictions = logistic_model.predict_proba(X_sentences)

		for i in range(len(sentences)):
			print(f"{max(min(100 - predictions[i][1]*50/TRESHOLD, 100), 0):.2f}% HATE:\t{sentences[i]}")

	elif type(sentences) is str:
		sentence = [sentences]
		X_sentence = [" ".join(preprocess(i)) for i in sentence]
		X_sentence = tfidf_vectorizer.transform(X_sentence).toarray()
		predictions = logistic_model.predict_proba(X_sentence)

		return max(min(100 - predictions[0][1]*50/TRESHOLD, 100), 0)

# Use from another script
def main(arg:str="Nothing"):

	if arg == "Nothing":
		print("use: main('file.txt') | main('my sentence')")
		exit(0)
	elif arg == "file" or (not ' ' in arg and arg[len(arg) - 4:] == '.txt'):
		with open(path1 + 'uploads/in.txt', 'r', encoding='utf-8') as file:
			content = file.read()
	else :
		content = arg

	res = predict_hate(content)
	
	return f"{res:2.2f}"

"""	
if __name__ == '__main__':
	print(main("hello everyone I'm lionel"))
"""
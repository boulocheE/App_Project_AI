from flask import Flask, request, render_template, jsonify
import os


from Natural_language_processing.nlp_from_scratch import *


app = Flask(__name__)

# dossier static : utilisation des images, js, css
app.static_folder = 'static'

# Dossier de téléchargement des images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# FICHIER HTML INDEX
@app.route('/')
def index() :
	return render_template('index.html')

@app.route('/pages/panneauxRoutiers')
def pagePanneauxRoutiers() :
	return render_template('pages/panneauxRoutiers.html')

@app.route('/pages/texte')
def pageTexte() :
	return render_template('pages/texte.html')





@app.route('/uploadPhoto', methods = ['POST'])
def uploadPhoto():
	# Vérifier si le formulaire contient une image
	if 'image' in request.files:
		fichier = request.files['image']


		# Vérifier si le fichier a un nom et est autorisé
		if fichier.filename != '' and fichier.filename.endswith(('.jpg', '.jpeg', '.png')):
			# Assurez-vous que le dossier d'upload existe, sinon créez-le
			if not os.path.exists(app.config['UPLOAD_FOLDER']):
				os.makedirs(app.config['UPLOAD_FOLDER'])

			# Enregistrez le fichier dans le dossier d'upload avec un nouveau nom si nécessaire
			chemin_enregistrement = os.path.join(app.config['UPLOAD_FOLDER'], 'in.png')
			fichier.save(chemin_enregistrement)
			return render_template ( 'pages/panneauxRoutiers.html', resultat = 'OK' )

	return render_template ( 'pages/panneauxRoutiers.html', erreur = 'Échec du téléchargement.' )



@app.route('/uploadFileTxt', methods = ['POST'])
def uploadFileTxt() :
	# Vérifier si le formulaire contient un fichier
	if 'file' in request.files:
		fichier = request.files['file']

		# Vérifier si le fichier a un nom et est autorisé
		if fichier.filename != '' and fichier.filename.endswith(('.txt')):
			# Assurez-vous que le dossier d'upload existe, sinon créez-le
			if not os.path.exists(app.config['UPLOAD_FOLDER']):
				os.makedirs(app.config['UPLOAD_FOLDER'])

			# Enregistrez le fichier dans le dossier d'upload avec un nouveau nom si nécessaire
			chemin_enregistrement = os.path.join(app.config['UPLOAD_FOLDER'], 'in.txt')
			fichier.save(chemin_enregistrement)

			res = main("file")

			return render_template ( 'pages/texte.html', resultat = res )


	return render_template ( 'pages/texte.html', resultat = "Error while downloading" )



@app.route('/uploadTxt', methods = ['POST'])
def uploadTxt() :
	if 'text' in request.form:
		texte = request.form['text']

		if len(texte.split()) <= 3 :
			return render_template('pages/texte.html', erreur = "Your text must contain at least 4 words. Try again.")

		if len(texte.split()) > 100 :
			return render_template('pages/texte.html', erreur = "Your text must contain less than 100 words. Try again.")


		res = main(texte)


		return render_template('pages/texte.html', resultat = res)

	return render_template( 'pages/texte.html', erreur = "Failure in the sending process. Try again." )



if __name__ == '__main__':
	app.run(debug=True)

from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Dossier de téléchargement des images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
	# Vérifier si le formulaire contient un fichier
	if 'image' in request.files:
		fichier = request.files['image']
		
		# Vérifier si le fichier a un nom et est autorisé
		if fichier.filename != '' and fichier.filename.endswith(('.jpg', '.jpeg', '.png')):
			# Assurez-vous que le dossier d'upload existe, sinon créez-le
			if not os.path.exists(app.config['UPLOAD_FOLDER']):
				os.makedirs(app.config['UPLOAD_FOLDER'])
			
			# Enregistrez le fichier dans le dossier d'upload avec un nouveau nom si nécessaire
			chemin_enregistrement = os.path.join(app.config['UPLOAD_FOLDER'], 'nouveau_nom_image.jpg')
			fichier.save(chemin_enregistrement)
			return 'Téléchargement réussi !'
	
	return 'Échec du téléchargement.'

if __name__ == '__main__':
	app.run(debug=True)

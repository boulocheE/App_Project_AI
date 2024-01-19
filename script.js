document.addEventListener('DOMContentLoaded', function() {
	var header = document.getElementById('header');

	var menu = document.getElementById('menu'); // Changer la position initiale selon vos besoins
	positionInitiale = header.offsetHeight;

	window.addEventListener('scroll', function() {
		var scrollTop = window.scrollY;

		if (scrollTop < positionInitiale)
			menu.style.top = positionInitiale - scrollTop + 'px';
		else
			menu.style.top = '0';
	});

});



function getValue() {
	// Sélectionner l'élément input et récupérer sa valeur
	var input = document.getElementById("in").value;

	// Afficher la valeur
	alert(input);
}



function changementEtat ( event, pageActuelle, pageSuivante, menu1, menu2 ) {
	pageSuivante = document.getElementById( pageSuivante );
	pageSuivante.style.display = 'block';

	pageActuelle = document.getElementById( pageActuelle );
	pageActuelle.style.display = 'none';


	menu1 = document.getElementById( menu1 );
	menu1.classList.remove("choix");

	menu2 = document.getElementById( menu2 );
	menu2.classList.add("choix");
}


function formulaireFichierTexte ( event, choix1, choix2 ) {
	choix2 = document.getElementById( choix2 );
	choix2.style.display = 'block';

	choix1 = document.getElementById( choix1 );
	choix1.style.display = 'none';
}
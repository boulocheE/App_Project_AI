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

const apiUrl = "https://rya2310.pythonanywhere.com/api/pokemon?name=";

const searchBox = document.querySelector(".search input");
const searchBtn = document.querySelector(".search button");


async function check(name) {
    const response = await fetch(apiUrl + name);
    var data = await response.json();
    
    document.querySelector(".pokemon").innerHTML = data[0].name;
    document.querySelector(".type").innerHTML = data[0].type;
    document.querySelector(".hp").innerHTML = data[0].hitpoints;
}

searchBtn.addEventListener('click',
    () => {
        check(searchBox.value);
    })
